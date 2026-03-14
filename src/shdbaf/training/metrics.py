from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from shdbaf.training.system import sync_if_needed


def save_confusion_matrix_png(cm: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    fig = plt.figure()
    plt.imshow(cm)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def evaluate_full(model: nn.Module, loader: DataLoader, device: str, labels: list[str]):
    """Évaluation complète (macro-F1, acc, CM, report)."""
    num_classes = len(labels)
    model.eval()
    y_true_all, y_pred_all = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        sync_if_needed(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        y_true_all.append(y.cpu().numpy())
        y_pred_all.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    acc = float((y_true == y_pred).mean())
    f1_macro = float(f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes))))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    rep = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return acc, f1_macro, cm, rep


@torch.no_grad()
def evaluate_partial(model: nn.Module, loader: DataLoader, device: str, labels: list[str], max_batches: int):
    """Évaluation rapide (signal live pendant l'epoch)."""
    num_classes = len(labels)
    model.eval()
    y_true_all, y_pred_all = [], []

    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)
        sync_if_needed(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        y_true_all.append(y.cpu().numpy())
        y_pred_all.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    acc = float((y_true == y_pred).mean())
    f1_macro = float(f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes))))
    return acc, f1_macro