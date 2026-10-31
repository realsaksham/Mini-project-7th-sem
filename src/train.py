"""
Single-run training entry point, shared by the baseline and the hybrid model.

Usage:
    python -m src.train --model baseline --subset   # fast debug run
    python -m src.train --model baseline            # full run
    python -m src.train --model hybrid   --subset
    python -m src.train --model hybrid

Both --model choices go through the exact same train_model()/evaluate_model()
code path (src/models.py's build_model() is the only thing that differs), so
the classical-vs-hybrid comparison is controlled.
"""

import argparse
import json
import os
import sys
import time

import pennylane as qml
import torch
import torch.nn as nn

from configs.config import get_config
from src.data import load_pneumoniamnist, set_seed
from src.evaluate import evaluate_model, save_confusion_matrix, save_metrics_json
from src.models import build_model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, val_loader, cfg, device="cpu"):
    """
    Standard PyTorch training loop. Works identically for the classical and
    quantum-head models -- it only ever calls model(images)/loss.backward(),
    it has no idea a quantum circuit is involved.

    Returns:
        history: list of per-epoch dicts (train_loss, train_acc, val_loss, val_acc)
        duration_sec: wall-clock training time
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCELoss()  # model already outputs a sigmoid probability

    history = []
    start = time.perf_counter()

    for epoch in range(cfg.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze(1).float().to(device)

            optimizer.zero_grad()
            probs = model(images).squeeze(1)
            loss = criterion(probs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += ((probs >= 0.5).float() == labels).sum().item()
            train_total += images.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.squeeze(1).float().to(device)
                probs = model(images).squeeze(1)
                loss = criterion(probs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += ((probs >= 0.5).float() == labels).sum().item()
                val_total += images.size(0)

        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_total, 1),
            "train_acc": train_correct / max(train_total, 1),
            "val_loss": val_loss / max(val_total, 1),
            "val_acc": val_correct / max(val_total, 1),
        }
        history.append(epoch_stats)
        print(f"  epoch {epoch_stats['epoch']}/{cfg.epochs} "
              f"train_loss={epoch_stats['train_loss']:.4f} "
              f"train_acc={epoch_stats['train_acc']:.4f} "
              f"val_loss={epoch_stats['val_loss']:.4f} "
              f"val_acc={epoch_stats['val_acc']:.4f}")

    duration_sec = time.perf_counter() - start
    return history, duration_sec


def print_comparison_table(results_dir: str) -> None:
    """If both baseline.json and hybrid.json exist, print a side-by-side table."""
    baseline_path = os.path.join(results_dir, "baseline.json")
    hybrid_path = os.path.join(results_dir, "hybrid.json")
    if not (os.path.exists(baseline_path) and os.path.exists(hybrid_path)):
        return

    with open(baseline_path) as f:
        b = json.load(f)
    with open(hybrid_path) as f:
        h = json.load(f)

    rows = ["accuracy", "precision", "recall", "f1", "roc_auc",
            "trainable_params", "train_time_sec"]
    print("\n=== Baseline vs. Hybrid ===")
    print(f"{'metric':<18}{'baseline':>14}{'hybrid':>14}")
    for row in rows:
        bv, hv = b.get(row), h.get(row)
        bv_s = f"{bv:.4f}" if isinstance(bv, float) else str(bv)
        hv_s = f"{hv:.4f}" if isinstance(hv, float) else str(hv)
        print(f"{row:<18}{bv_s:>14}{hv_s:>14}")


def run(model_type: str, subset: bool) -> None:
    cfg = get_config(subset=subset)
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)

    head_type = "classical" if model_type == "baseline" else "quantum"
    print(f"[{model_type}] config: epochs={cfg.epochs} batch_size={cfg.batch_size} "
          f"subset={subset} n_qubits={cfg.n_qubits} depth={cfg.circuit_depth}")

    train_loader, val_loader, test_loader = load_pneumoniamnist(cfg, subset=subset)
    model = build_model(cfg, head_type=head_type)

    n_params = count_trainable_params(model)
    print(f"[{model_type}] trainable parameters: {n_params}")

    history, duration_sec = train_model(model, train_loader, val_loader, cfg)
    print(f"[{model_type}] training time: {duration_sec:.2f}s")

    metrics, labels, preds = evaluate_model(model, test_loader)
    metrics["trainable_params"] = n_params
    metrics["train_time_sec"] = duration_sec
    metrics["history"] = history
    metrics["config"] = {
        "epochs": cfg.epochs, "batch_size": cfg.batch_size, "subset": subset,
        "n_qubits": cfg.n_qubits, "circuit_depth": cfg.circuit_depth,
        "lr": cfg.lr, "seed": cfg.seed,
    }

    print(f"[{model_type}] test metrics: "
          f"acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
          f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
          f"roc_auc={metrics['roc_auc']:.4f}")

    out_name = "baseline" if model_type == "baseline" else "hybrid"
    save_metrics_json(metrics, os.path.join(cfg.results_dir, f"{out_name}.json"))
    save_confusion_matrix(
        labels, preds,
        os.path.join(cfg.results_dir, f"confusion_matrix_{out_name}.png"),
        title=f"{model_type.capitalize()} Confusion Matrix",
    )

    if model_type == "hybrid":
        # qml.draw() renders the circuit's gate layout as ASCII art -- useful
        # to drop straight into a report/slide and to sanity-check that the
        # AngleEmbedding/StronglyEntanglingLayers/measurement structure is
        # what we intended. Needs one concrete input to trace through.
        dummy_inputs = torch.zeros(cfg.n_qubits)
        dummy_weights = torch.zeros(
            qml.StronglyEntanglingLayers.shape(n_layers=cfg.circuit_depth, n_wires=cfg.n_qubits)
        )
        diagram = qml.draw(model.head.circuit)(dummy_inputs, dummy_weights)
        # qml.draw's output uses unicode box-drawing / bra-ket characters that
        # Windows' default console codepage (cp1252) can't print -- fall back
        # to a safe re-encode for the terminal, but always keep the full
        # unicode diagram in the saved file.
        safe_diagram = diagram.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"
        )
        print("\n" + safe_diagram)
        with open(os.path.join(cfg.results_dir, "circuit_diagram.txt"), "w", encoding="utf-8") as f:
            f.write(diagram + "\n")

    print_comparison_table(cfg.results_dir)


def main():
    parser = argparse.ArgumentParser(description="Train the baseline or hybrid disease classifier.")
    parser.add_argument("--model", choices=["baseline", "hybrid"], required=True)
    parser.add_argument("--subset", action="store_true",
                         help="Train on a small subset for fast debugging (~200 images, 2 epochs).")
    args = parser.parse_args()
    run(args.model, args.subset)


if __name__ == "__main__":
    main()
