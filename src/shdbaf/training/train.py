from __future__ import annotations

import warnings
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shdbaf.artifacts import (
    artifacts_dir,
    manifest_parquet,
    mlflow_db,
    processed_dataset_dir,
)
from shdbaf.data.torch_dataset import MemmapDatasetConfig, MemmapWindowDataset
from shdbaf.models.resnet1d import ResNet1D
from shdbaf.settings import load_settings
from shdbaf.training.loop import train_one_epoch
from shdbaf.training.metrics import evaluate_full, save_confusion_matrix_png
from shdbaf.training.mlflow_utils import (
    init_experiment,
    log_and_register_torch_model,
    log_json,
    log_text,
    set_tracking_uri_sqlite,
)
from shdbaf.training.system import get_device

warnings.filterwarnings("ignore", message="gmpy2 version is too old to use")


def compute_class_counts(manifest_path: Path) -> dict:
    m = pl.read_parquet(manifest_path)
    out: dict = {}
    for split in ["train", "val", "test"]:
        ms = m.filter(pl.col("split") == split)
        d = ms.group_by(["label", "label_id"]).len().sort("label_id").to_dicts()
        out[split] = {row["label"]: int(row["len"]) for row in d}
    return out


def compute_class_weights(
    manifest_path: Path, split: str, num_classes: int
) -> np.ndarray:
    m = pl.read_parquet(manifest_path).filter(pl.col("split") == split)
    counts = m.group_by("label_id").len().sort("label_id")["len"].to_numpy()
    if len(counts) < num_classes:
        counts = np.pad(counts, (0, num_classes - len(counts)), constant_values=0)
    w = 1.0 / np.maximum(counts, 1)
    w = w / w.sum() * num_classes
    return w.astype(np.float32)


def safe_torch_load(path: Path, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    S = load_settings()

    torch.manual_seed(S.seed)
    np.random.seed(S.seed)

    # dataset paths
    tag = S.dataset.tag()
    root = processed_dataset_dir(tag)
    manifest = manifest_parquet(tag)
    if not manifest.exists():
        raise FileNotFoundError(
            f"manifest introuvable: {manifest}\n→ Lance `pixi run make-windows` (tag={tag})."
        )

    labels = list(S.dataset.classes)
    num_classes = len(labels)

    # params train
    batch_size = int(S.train.batch_size)
    lr = float(S.train.lr)
    epochs = int(S.train.epochs)
    steps_per_epoch = int(S.train.steps_per_epoch)
    num_workers = int(S.train.num_workers)
    cache_size = int(S.train.cache_size)
    shuffle = bool(S.train.shuffle)
    log_every = int(S.train.log_every)

    eval_every = int(getattr(S.train, "eval_every", 0))
    eval_batches = int(getattr(S.train, "eval_batches", 0))

    # model params
    base = int(S.model.base)
    k = int(S.model.k)
    blocks = tuple(int(x) for x in S.model.blocks)

    device = get_device()

    # MLflow
    tracking_uri = set_tracking_uri_sqlite(mlflow_db())
    init_experiment("shdb-af-ecg")

    run_name = (
        f"{S.model.name}_{tag}_bs{batch_size}_lr{lr}_shuffle{int(shuffle)}_seed{S.seed}"
    )

    # artifacts
    art_dir = artifacts_dir()
    art_dir.mkdir(parents=True, exist_ok=True)

    best_path = art_dir / "best.pt"
    counts_path = art_dir / "class_counts.json"
    weights_path = art_dir / "class_weights.json"

    cm_val_txt = art_dir / "cm_val.txt"
    cm_val_png = art_dir / "cm_val.png"
    rep_val_json = art_dir / "report_val.json"

    cm_test_txt = art_dir / "cm_test.txt"
    cm_test_png = art_dir / "cm_test.png"
    rep_test_json = art_dir / "report_test.json"

    # Registry name (stable)
    registered_model_name = f"shdbaf_{S.model.name}_{len(labels)}c"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "seed": S.seed,
                "device": device,
                "dataset_tag": tag,
                "data_dir": str(root),
                "manifest": str(manifest),
                "labels": ",".join(labels),
                "batch_size": batch_size,
                "lr": lr,
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "num_workers": num_workers,
                "cache_size": cache_size,
                "shuffle": shuffle,
                "log_every": log_every,
                "eval_every": eval_every,
                "eval_batches": eval_batches,
                "model": S.model.name,
                "base": base,
                "k": k,
                "blocks": str(blocks),
                "tracking_uri": tracking_uri,
                "registered_model_name": registered_model_name,
            }
        )

        # counts/weights
        counts = compute_class_counts(manifest)
        log_json(counts_path, counts, artifact_path="data")

        cls_w = compute_class_weights(manifest, "train", num_classes)
        weights_dict = {labels[i]: float(cls_w[i]) for i in range(num_classes)}
        log_json(weights_path, weights_dict, artifact_path="data")

        # datasets + loaders
        train_ds = MemmapWindowDataset(
            MemmapDatasetConfig(
                root=root, manifest_path=manifest, split="train", cache_size=cache_size
            )
        )
        val_ds = MemmapWindowDataset(
            MemmapDatasetConfig(
                root=root, manifest_path=manifest, split="val", cache_size=cache_size
            )
        )
        test_ds = MemmapWindowDataset(
            MemmapDatasetConfig(
                root=root, manifest_path=manifest, split="test", cache_size=cache_size
            )
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=False,
        )
        # reco macOS: val/test num_workers=0
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # model/loss/opt
        model = ResNet1D(
            in_ch=2, num_classes=num_classes, base=base, blocks=blocks, k=k
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(cls_w, dtype=torch.float32, device=device)
        )

        global_step = 0
        best_val_f1 = -1.0

        # epochs
        for epoch in range(epochs):
            res = train_one_epoch(
                epoch=epoch,
                model=model,
                opt=opt,
                loss_fn=loss_fn,
                train_loader=train_loader,
                val_loader=val_loader,
                labels=labels,
                device=device,
                batch_size=batch_size,
                steps_per_epoch=steps_per_epoch,
                log_every=log_every,
                eval_every=eval_every,
                eval_batches=eval_batches,
                global_step=global_step,
            )
            global_step = res.global_step

            # validation complète
            val_acc, val_f1, cm_val, rep_val = evaluate_full(
                model, val_loader, device, labels
            )

            # F1 par classe
            val_f1_per_class = {
                f"val_f1_{lab}": float(rep_val.get(lab, {}).get("f1-score", 0.0))
                for lab in labels
            }
            mlflow.log_metrics(val_f1_per_class, step=epoch)

            print(
                f"[epoch {epoch + 1}] done in {res.epoch_duration_s:.1f}s "
                f"train_loss={res.train_loss_epoch:.4f} val_f1_macro={val_f1:.4f} val_acc={val_acc:.4f} "
                f"sps_epoch={res.samples_per_sec_epoch:.0f}/s"
            )

            mlflow.log_metrics(
                {
                    "epoch_duration_s": res.epoch_duration_s,
                    "train_loss_epoch": res.train_loss_epoch,
                    "val_f1_macro": float(val_f1),
                    "val_acc": float(val_acc),
                    "samples_per_sec_epoch": res.samples_per_sec_epoch,
                },
                step=epoch,
            )

            # best
            if val_f1 > best_val_f1:
                best_val_f1 = float(val_f1)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_f1": float(val_f1),
                    },
                    best_path,
                )
                mlflow.log_artifact(str(best_path), artifact_path="checkpoints")

                log_text(cm_val_txt, str(cm_val), artifact_path="eval")
                log_json(rep_val_json, rep_val, artifact_path="eval")
                save_confusion_matrix_png(
                    cm_val, labels, cm_val_png, title="Confusion matrix (val)"
                )
                mlflow.log_artifact(str(cm_val_png), artifact_path="eval")

        # test on best + register model
        if best_path.exists():
            state = safe_torch_load(best_path, device)
            model.load_state_dict(state["model_state"])

        # Register best model in the Model Registry
        model_tags = {
            "dataset_tag": tag,
            "seed": str(S.seed),
            "win_sec": str(S.dataset.win_sec),
            "stride_sec": str(S.dataset.stride_sec),
            "majority_thr": str(S.dataset.majority_thr),
            "channels": str(S.dataset.channels),
            "classes": ",".join(labels),
        }
        model_uri = log_and_register_torch_model(
            model,
            artifact_path="model",
            registered_model_name=registered_model_name,
            tags=model_tags,
        )
        mlflow.log_param("registered_model_uri", model_uri)

        # IMPORTANT: remettre le modèle sur le device pour la suite (test)
        model.to(device).eval()

        # Final test eval
        test_acc, test_f1, cm_test, rep_test = evaluate_full(
            model, test_loader, device, labels
        )

        test_f1_per_class = {
            f"test_f1_{lab}": float(rep_test.get(lab, {}).get("f1-score", 0.0))
            for lab in labels
        }
        mlflow.log_metrics(test_f1_per_class)

        mlflow.log_metrics(
            {
                "best_val_f1_macro": float(best_val_f1),
                "test_f1_macro": float(test_f1),
                "test_acc": float(test_acc),
            }
        )

        log_text(cm_test_txt, str(cm_test), artifact_path="eval")
        log_json(rep_test_json, rep_test, artifact_path="eval")
        save_confusion_matrix_png(
            cm_test, labels, cm_test_png, title="Confusion matrix (test)"
        )
        mlflow.log_artifact(str(cm_test_png), artifact_path="eval")


if __name__ == "__main__":
    main()
