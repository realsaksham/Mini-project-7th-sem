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
                              precision_score, recall_score, roc_auc_score, roc_curve)


@torch.no_grad()
def predict_probs(model, loader, device="cpu"):
    """
    Run `model` over every batch in `loader` (in loader order) and return
    (probs, labels) as numpy arrays -- probs is P(positive class) per image.
    """
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        probs = model(images).squeeze(1)  # (batch,) probability of the positive class
        all_probs.append(probs.cpu())
        all_labels.append(labels.squeeze(1).float())

    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def evaluate_model(model, loader, device="cpu") -> dict:
    """
    Run `model` over every batch in `loader` and compute the standard binary
    classification metrics. Returns labels/predictions too (not just
    scalars) so callers can also build a confusion matrix without a second
    pass over the data.
    """
    probs, labels = predict_probs(model, loader, device)
    return compute_metrics(probs, labels)


def compute_metrics(probs, labels):
    """Metrics from precomputed probabilities; returns (metrics, labels, preds)."""
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


def save_confusion_matrix(labels, preds, path: str, title: str = "Confusion Matrix",
                          class_names=("Normal", "Positive")) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        labels, preds, display_labels=list(class_names),
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


def save_roc_curves(curves: dict, path: str, title: str = "ROC curves") -> None:
    """
    curves: {label: (labels, probs, style_kwargs)} -> one ROC plot, all
    models on the same axes, AUC in each legend entry.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (labels, probs, style) in curves.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC {roc_auc_score(labels, probs):.4f})", **style)
    ax.plot([0, 1], [0, 1], color="#b9b8b2", lw=1, ls=":", label="chance")
    ax.set_xlabel("False positive rate", color="#52514e")
    ax.set_ylabel("True positive rate (recall)", color="#52514e")
    ax.set_title(title, loc="left", color="#0b0b0b")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
