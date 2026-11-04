"""
Main hybrid study: classical vs. classical->quantum vs. quantum->classical,
each next to a fair classical control, across training-set sizes.

    model                         what it tests
    ----------------------------  ------------------------------------------
    Classical CNN                 baseline: CNN + Linear head
    CNN + matched classical head  CONTROL for C->Q: same bottleneck as the
                                  quantum head, quantum layer -> Linear(4,4)+tanh
    C->Q hybrid                   CNN features -> variational quantum circuit
    Q->C hybrid                   quanvolution on pixels -> classical CNN
    Random patch + CNN            CONTROL for Q->C: fixed random classical
                                  patch map in place of the quantum filter

Low-data: every model is trained on 2%, 5%, 10%, 25% and 100% of the training
split (stratified), 3 seeds each, and always scored on the same full
validation/test sets. Small fractions train for more epochs (same rule for
every model) and every run keeps its best-validation-loss epoch, so no model
is handicapped by under- or over-training.

Fairness rules: identical data, splits, seeds, optimizer, epochs, early
stopping and evaluation for every model; no per-model tuning.

Significance: on the full-data runs, McNemar's exact test compares each
hybrid with its control on the same 630 test images (paired), for every seed.

Usage:
    python -m src.hybrid_study --subset        # fast debug run
    python -m src.hybrid_study                 # full study
"""

import argparse
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from configs.config import DATASET_PRESETS, get_config
from src.data import load_data, set_seed
from src.evaluate import compute_metrics, predict_probs
from src.models import build_model
from src.train import count_trainable_params, train_model

MODELS = [
    # (name, head_type, input_transform)
    ("Classical CNN", "classical", "none"),
    ("CNN + matched classical head", "classical_matched", "none"),
    ("C->Q hybrid (CNN + quantum head)", "quantum", "none"),
    ("Q->C hybrid (quanvolution + CNN)", "classical", "quanv"),
    ("Random patch + CNN (control)", "classical", "random_patch"),
    ("Avg-pool + CNN (control)", "classical", "avgpool"),
    ("Learned 2x2 conv + CNN (control)", "classical", "learned_patch"),
]
# Each hybrid paired with the classical control that isolates its quantum part.
PAIRS = [
    ("C->Q hybrid (CNN + quantum head)", "CNN + matched classical head"),
    ("Q->C hybrid (quanvolution + CNN)", "Random patch + CNN (control)"),
    ("Q->C hybrid (quanvolution + CNN)", "Avg-pool + CNN (control)"),
    ("Q->C hybrid (quanvolution + CNN)", "Learned 2x2 conv + CNN (control)"),
    ("C->Q hybrid (CNN + quantum head)", "Classical CNN"),
    ("Q->C hybrid (quanvolution + CNN)", "Classical CNN"),
]
FRACTIONS = [0.02, 0.05, 0.1, 0.25, 1.0]
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")

# Categorical slots 1-5 of the dataviz skill's validated palette, fixed order.
# Controls are dashed so identity never rests on colour alone.
STYLES = {
    "Classical CNN": dict(color="#2a78d6", ls="-", marker="o"),
    "CNN + matched classical head": dict(color="#eb6834", ls="--", marker="s"),
    "C->Q hybrid (CNN + quantum head)": dict(color="#1baf7a", ls="-", marker="D"),
    "Q->C hybrid (quanvolution + CNN)": dict(color="#eda100", ls="-", marker="^"),
    "Random patch + CNN (control)": dict(color="#e87ba4", ls="--", marker="v"),
    "Avg-pool + CNN (control)": dict(color="#008300", ls="--", marker="P"),
    "Learned 2x2 conv + CNN (control)": dict(color="#6250d6", ls="--", marker="X"),
}
INK, MUTED_INK = "#0b0b0b", "#52514e"


def epochs_for(fraction: float, base_epochs: int) -> int:
    """Same rule for every model: tiny training sets get more passes."""
    return base_epochs if fraction >= 0.25 else 40


def mcnemar_p(wrong_a: np.ndarray, wrong_b: np.ndarray) -> float:
    """Exact McNemar test on paired per-image correctness."""
    b = int(np.sum(~wrong_a & wrong_b))  # A right, B wrong
    c = int(np.sum(wrong_a & ~wrong_b))  # A wrong, B right
    if b + c == 0:
        return 1.0
    return binomtest(b, b + c, 0.5).pvalue


def plot_vs_fraction(summary, metric, ylabel, path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for name, _, _ in MODELS:
        g = summary[summary["model"] == name].sort_values("train_fraction")
        x = g["train_fraction"] * 100
        ax.plot(x, g[f"{metric}_mean"], lw=2, ms=7, label=name, **STYLES[name])
        # Metrics live in [0, 1]; clip the +-std band so it never implies otherwise.
        ax.fill_between(x, (g[f"{metric}_mean"] - g[f"{metric}_std"]).clip(0, 1),
                        (g[f"{metric}_mean"] + g[f"{metric}_std"]).clip(0, 1),
                        color=STYLES[name]["color"], alpha=0.10, lw=0)
    ax.set_xscale("log")
    ax.set_xticks([f * 100 for f in FRACTIONS])
    ax.set_xticklabels([f"{f * 100:g}%" for f in FRACTIONS])
    ax.set_xlabel("Training data used (log scale)", color=MUTED_INK)
    ax.set_ylabel(ylabel, color=MUTED_INK)
    ax.set_title(f"{ylabel} vs. training-set size (mean ± std, 3 seeds)", loc="left", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.grid(True, alpha=0.25)
    if metric == "f1":
        ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def run(subset: bool, dataset: str, seeds, fractions) -> None:
    base_cfg = get_config(subset=subset, dataset=dataset)
    out_dir = os.path.join(base_cfg.results_dir, "hybrid_study")
    os.makedirs(out_dir, exist_ok=True)

    rows, wrong_full = [], {}  # wrong_full[(model, seed)] -> per-test-image error mask
    all_test_probs = {}         # full-data test probabilities, saved for later re-analysis
    for fraction in fractions:
        for name, head, transform in MODELS:
            for seed in seeds:
                cfg = replace(base_cfg, seed=seed, train_fraction=fraction,
                              input_transform=transform, restore_best=True,
                              epochs=base_cfg.epochs if subset else epochs_for(fraction, base_cfg.epochs))
                set_seed(seed)
                print(f"\n=== {fraction:.0%} data | {name} | seed {seed} ===")
                train_loader, val_loader, test_loader = load_data(cfg, subset=subset)
                model = build_model(cfg, head_type=head)
                n_params = count_trainable_params(model)
                head_params = count_trainable_params(model.head)

                _, duration_sec = train_model(model, train_loader, val_loader, cfg)
                val_probs, val_labels = predict_probs(model, val_loader)
                test_probs, test_labels = predict_probs(model, test_loader)
                val_m, _, _ = compute_metrics(val_probs, val_labels)
                test_m, _, preds = compute_metrics(test_probs, test_labels)
                if fraction == fractions[-1]:
                    wrong_full[(name, seed)] = preds != test_labels
                    all_test_probs[f"{name}|seed{seed}"] = test_probs

                row = {"model": name, "train_fraction": fraction, "seed": seed,
                       "n_train": len(train_loader.dataset), "epochs": cfg.epochs}
                row.update({f"val_{m}": val_m[m] for m in METRICS})
                row.update({m: test_m[m] for m in METRICS})
                row.update(trainable_params=n_params, head_params=head_params,
                           train_time_sec=duration_sec)
                rows.append(row)
                print(f"-> n_train={row['n_train']} acc={test_m['accuracy']:.4f} "
                      f"f1={test_m['f1']:.4f} auc={test_m['roc_auc']:.4f} time={duration_sec:.1f}s")

        # Save after every fraction so a long run can be inspected part-way.
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "runs.csv"), index=False)

    np.savez_compressed(os.path.join(out_dir, "test_probs_full_data.npz"), **all_test_probs)
    runs = pd.DataFrame(rows)
    cols = list(METRICS) + ["val_f1", "train_time_sec"]
    summary = runs.groupby(["model", "train_fraction"], sort=False)[cols].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()
    firsts = runs.groupby("model")[["trainable_params", "head_params"]].first()
    summary = summary.join(firsts, on="model")
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # --- markdown report ---
    def pm(r, c):
        std = r[f"{c}_std"]
        return f"{r[f'{c}_mean']:.4f} ± {std:.4f}" if not np.isnan(std) else f"{r[f'{c}_mean']:.4f}"

    md = ["# Hybrid study: classical vs. C->Q vs. Q->C", "",
          f"Seeds {', '.join(map(str, seeds))}; mean ± std over seeds; "
          "same fixed validation/test split for every run.", ""]
    for fraction in fractions:
        n_train = int(runs[runs["train_fraction"] == fraction]["n_train"].iloc[0])
        md += [f"## {fraction:.0%} of training data ({n_train} images)", "",
               "| model | accuracy | recall | F1 | ROC-AUC | params (head) | time (s) |",
               "|---|---|---|---|---|---|---|"]
        for _, r in summary[summary["train_fraction"] == fraction].iterrows():
            md.append(f"| {r['model']} | {pm(r, 'accuracy')} | {pm(r, 'recall')} | {pm(r, 'f1')} | "
                      f"{pm(r, 'roc_auc')} | {int(r['trainable_params']):,} ({int(r['head_params'])}) | "
                      f"{r['train_time_sec_mean']:.0f} |")
        md.append("")

    md += [f"## Significance at {fractions[-1]:.0%} data (McNemar exact test, paired test images)", "",
           "p < 0.05 means the two models' error patterns differ beyond chance for that seed.", "",
           "| hybrid | vs. | " + " | ".join(f"p (seed {s})" for s in seeds) + " |",
           "|---|---|" + "---|" * len(seeds)]
    for a, b in PAIRS:
        ps = [mcnemar_p(wrong_full[(a, s)], wrong_full[(b, s)]) for s in seeds]
        md.append(f"| {a} | {b} | " + " | ".join(f"{p:.3f}" for p in ps) + " |")
    md.append("")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n" + "\n".join(md))

    if len(fractions) > 1:
        plot_vs_fraction(summary, "f1", "Test F1", os.path.join(out_dir, "f1_vs_data.png"))
        plot_vs_fraction(summary, "roc_auc", "Test ROC-AUC", os.path.join(out_dir, "auc_vs_data.png"))
        plot_vs_fraction(summary, "accuracy", "Test accuracy", os.path.join(out_dir, "accuracy_vs_data.png"))


def main():
    parser = argparse.ArgumentParser(description="Classical vs. hybrid study with low-data sweep.")
    parser.add_argument("--subset", action="store_true", help="Fast debug run.")
    parser.add_argument("--dataset", choices=list(DATASET_PRESETS), default="tb")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    args = parser.parse_args()
    run(args.subset, args.dataset, args.seeds, args.fractions)


if __name__ == "__main__":
    main()
