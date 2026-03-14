from __future__ import annotations

import random
from dataclasses import dataclass

import polars as pl

from shdbaf.artifacts import additional_csv, index_parquet, splits_parquet
from shdbaf.settings import load_settings


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    only_annotated: bool


def make_subject_split_map(subject_ids: list[str], cfg: SplitConfig) -> dict[str, str]:
    """Mapping déterministe subject_id -> split (par sujet, donc sans fuite)."""
    assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 1e-9

    rng = random.Random(cfg.seed)
    subs = subject_ids.copy()
    rng.shuffle(subs)

    n = len(subs)
    n_train = int(round(cfg.train_ratio * n))
    n_val = int(round(cfg.val_ratio * n))
    # le reste en test pour éviter les soucis d'arrondi
    train = set(subs[:n_train])
    val = set(subs[n_train : n_train + n_val])
    test = set(subs[n_train + n_val :])

    mapping: dict[str, str] = {}
    for s in train:
        mapping[s] = "train"
    for s in val:
        mapping[s] = "val"
    for s in test:
        mapping[s] = "test"
    return mapping


def main() -> None:
    S = load_settings()

    cfg = SplitConfig(
        seed=S.seed,
        train_ratio=S.split.train_ratio,
        val_ratio=S.split.val_ratio,
        test_ratio=S.split.test_ratio,
        only_annotated=S.split.only_annotated,
    )

    index_path = index_parquet()
    additional_path = additional_csv()
    out_path = splits_parquet()

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index manquant: {index_path} (lance `pixi run index`)."
        )
    if not additional_path.exists():
        raise FileNotFoundError(
            f"AdditionalData.csv introuvable: {additional_path} (lance `pixi run download`)."
        )

    idx = pl.read_parquet(index_path)
    add = pl.read_csv(additional_path, infer_schema_length=1000)

    add = add.select(
        pl.col("Data_ID").cast(pl.Int64).alias("data_id"),
        pl.col("Subject_ID").cast(pl.Utf8).alias("subject_id"),
        pl.col("Annotated").cast(pl.Int8).alias("annotated_flag"),
    ).with_columns(pl.col("data_id").cast(pl.Utf8).str.zfill(3).alias("record"))

    df = idx.join(
        add.select(["record", "subject_id", "annotated_flag"]), on="record", how="left"
    )

    missing = (
        df.filter(pl.col("subject_id").is_null()).select(["record", "path"]).head(20)
    )
    if missing.height:
        n_missing = df.filter(pl.col("subject_id").is_null()).height
        raise RuntimeError(
            f"{n_missing} records n'ont pas de subject_id après join.\n"
            f"Exemples:\n{missing}\n"
            f"→ Vérifie que tes fichiers WFDB sont bien 001..128 et que AdditionalData.csv correspond."
        )

    if cfg.only_annotated:
        df = df.filter((pl.col("has_atr") == True) & (pl.col("annotated_flag") == 1))

    subjects = df.select(pl.col("subject_id").unique()).to_series().to_list()
    mapping = make_subject_split_map(subjects, cfg)
    map_df = pl.DataFrame(
        {"subject_id": list(mapping.keys()), "split": list(mapping.values())}
    )
    df = df.join(map_df, on="subject_id", how="left")

    if df.filter(pl.col("split").is_null()).height:
        raise RuntimeError("Certains sujets n'ont pas reçu de split (bug mapping).")

    leaks = (
        df.group_by("subject_id")
        .agg(pl.col("split").n_unique().alias("n_splits"))
        .filter(pl.col("n_splits") > 1)
    )
    if leaks.height:
        raise RuntimeError(
            f"Fuite détectée (même sujet dans plusieurs splits):\n{leaks}"
        )

    out = df.select(
        "record",
        "subject_id",
        "split",
        "has_atr",
        "fs",
        "n_sig",
        "sig_len",
        "duration_sec",
    ).sort(["split", "subject_id", "record"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(out_path)

    print(f"✅ Wrote {out_path.resolve()} ({out.height} records)")
    print(
        f"seed={cfg.seed} ratios train/val/test={cfg.train_ratio}/{cfg.val_ratio}/{cfg.test_ratio} only_annotated={cfg.only_annotated}"
    )
    print(out.group_by("split").len().sort("split"))


if __name__ == "__main__":
    main()
