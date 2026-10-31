"""
Evaluation utilities shared by the baseline and hybrid model, so both are
scored with exactly the same code path -- any metric difference between them
reflects the model, not the measurement.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display needed to save PNGs
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                              precision_score, recall_score, roc_auc_score)


@torch.no_grad()
def evaluate_model(model, loader, device="cpu") -> dict:
    """
    Run `model` over every batch in `loader` and compute the standard binary
    classification metrics. Returns probabilities/labels too (not just
    scalars) so callers can also build a confusion matrix without a second
    pass over the data.
    """
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        probs = model(images).squeeze(1)  # (batch,) probability of "pneumonia"
        all_probs.append(probs.cpu())
        all_labels.append(labels.squeeze(1).float())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        # roc_auc_score needs both classes present in `labels`; guard the
        # tiny --subset case where that might not hold.
        "roc_auc": roc_auc_score(labels, probs) if len(set(labels.tolist())) > 1 else float("nan"),
    }
    return metrics, labels, preds


def save_confusion_matrix(labels, preds, path: str, title: str = "Confusion Matrix") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        labels, preds, display_labels=["Normal", "Pneumonia"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_metrics_json(metrics: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
