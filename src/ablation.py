"""
Small ablation: hybrid model accuracy (and other metrics) vs. circuit depth,
depth in {1, 2, 3}, everything else held fixed at configs.config defaults.

This is intentionally NOT the full ablation study (that's a later stage) --
just one swept hyperparameter, reusing src.train's train_model()/evaluate_model
code path so results are directly comparable to the main hybrid run.

Usage:
    python -m src.ablation --subset   # fast debug run
    python -m src.ablation            # full run
"""

import argparse
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from configs.config import get_config
from src.data import load_pneumoniamnist, set_seed
from src.evaluate import evaluate_model
from src.models import build_model
from src.train import count_trainable_params, train_model

DEPTHS = [1, 2, 3]


def run(subset: bool) -> None:
    base_cfg = get_config(subset=subset)
    os.makedirs(base_cfg.results_dir, exist_ok=True)

    rows = []
    for depth in DEPTHS:
        # Re-seed before every depth so each run starts from the same data
        # split/shuffle and weight-init conditions -- depth is the only
        # thing that changes between rows.
        set_seed(base_cfg.seed)
        cfg = replace(base_cfg, circuit_depth=depth)

        print(f"\n=== ablation: circuit_depth={depth} ===")
        train_loader, val_loader, test_loader = load_pneumoniamnist(cfg, subset=subset)
        model = build_model(cfg, head_type="quantum")
        n_params = count_trainable_params(model)

        _, duration_sec = train_model(model, train_loader, val_loader, cfg)
        metrics, _, _ = evaluate_model(model, test_loader)

        row = {
            "depth": depth,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "trainable_params": n_params,
            "train_time_sec": duration_sec,
        }
        print(f"depth={depth} -> {row}")
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(base_cfg.results_dir, "ablation_depth.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(df["depth"], df["accuracy"], marker="o")
    ax.set_xlabel("Circuit depth (StronglyEntanglingLayers repetitions)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Hybrid model: accuracy vs. circuit depth")
    ax.set_xticks(DEPTHS)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(base_cfg.results_dir, "ablation_depth.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Hybrid model circuit-depth ablation (depth 1/2/3).")
    parser.add_argument("--subset", action="store_true",
                         help="Run on a small subset for fast debugging.")
    args = parser.parse_args()
    run(args.subset)


if __name__ == "__main__":
    main()
