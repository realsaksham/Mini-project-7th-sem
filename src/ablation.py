"""
Stage-3 ablation: how does the quantum head's configuration affect the hybrid
model? Sweeps, each varying one thing from the default hybrid config
(angle encoding, StronglyEntanglingLayers, 4 qubits, depth 2):

    depth         circuit_depth in {1, 2, 3, 4}
    entanglement  "strong" vs. "basic" vs. "none" (no CNOTs)
    qubits        n_qubits in {2, 4, 6, 8}, for BOTH angle and amplitude
                  encoding -- this sweep is also the encoding experiment

Every run reuses src.train's train_model()/evaluate_model() code path and is
re-seeded identically, so the swept setting is the only thing that differs.
Configs shared between sweeps (e.g. the default config) are trained once.

The "best" configuration is chosen on VALIDATION F1, not test metrics, so the
test set stays an unbiased estimate for the chosen config.

Usage:
    python -m src.ablation --subset                    # fast debug run
    python -m src.ablation                             # all sweeps
    python -m src.ablation --sweeps depth qubits       # only some sweeps
    python -m src.ablation --dataset pneumoniamnist
"""

import argparse
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from configs.config import DATASET_PRESETS, get_config
from src.data import load_data, set_seed
from src.evaluate import evaluate_model
from src.models import build_model
from src.train import count_trainable_params, train_model

# Each sweep is a list of overrides applied to the base config.
SWEEPS = {
    "depth": [dict(circuit_depth=d) for d in (1, 2, 3, 4)],
    "entanglement": [dict(entanglement=e) for e in ("strong", "basic", "none")],
    "qubits": [dict(encoding=enc, n_qubits=q)
               for enc in ("angle", "amplitude") for q in (2, 4, 6, 8)],
}

KEY_FIELDS = ("n_qubits", "circuit_depth", "encoding", "entanglement")
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")

# Categorical slots 1 and 2 of the dataviz skill's validated default palette.
ENCODING_COLORS = {"angle": "#2a78d6", "amplitude": "#eb6834"}
INK, MUTED_INK = "#0b0b0b", "#52514e"


def run_one(cfg, subset: bool) -> dict:
    set_seed(cfg.seed)
    train_loader, val_loader, test_loader = load_data(cfg, subset=subset)
    model = build_model(cfg, head_type="quantum")
    n_params = count_trainable_params(model)

    _, duration_sec = train_model(model, train_loader, val_loader, cfg)
    val_metrics, _, _ = evaluate_model(model, val_loader)
    test_metrics, _, _ = evaluate_model(model, test_loader)

    row = {k: getattr(cfg, k) for k in KEY_FIELDS}
    row.update({f"val_{m}": val_metrics[m] for m in METRICS})
    row.update({m: test_metrics[m] for m in METRICS})
    row.update(trainable_params=n_params, train_time_sec=duration_sec)
    return row


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=MUTED_INK)
    ax.set_ylabel(ylabel, color=MUTED_INK)
    ax.set_title(title, color=INK, loc="left")
    ax.grid(True, axis="y", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED_INK)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_depth(df, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(df["circuit_depth"], df["accuracy"], marker="o", lw=2, ms=8,
            color=ENCODING_COLORS["angle"])
    ax.set_xticks(df["circuit_depth"])
    _style(ax, "Circuit depth (layer repetitions)", "Test accuracy",
           "Accuracy vs. circuit depth")
    _save(fig, os.path.join(out_dir, "ablation_depth.png"))


def plot_qubits(df, out_dir):
    for y, ylabel, title, name in [
        ("accuracy", "Test accuracy", "Accuracy vs. qubits", "ablation_qubits.png"),
        ("train_time_sec", "Training time (s)", "Training time vs. qubits", "ablation_time_qubits.png"),
    ]:
        fig, ax = plt.subplots(figsize=(5, 4))
        for enc, g in df.groupby("encoding"):
            ax.plot(g["n_qubits"], g[y], marker="o", lw=2, ms=8,
                    color=ENCODING_COLORS[enc], label=f"{enc} encoding")
            # Direct label at the line end, in ink (colour carries identity).
            last = g.iloc[-1]
            ax.annotate(enc, (last["n_qubits"], last[y]), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK)
        ax.set_xticks(sorted(df["n_qubits"].unique()))
        ax.legend(frameon=False, labelcolor=INK)
        _style(ax, "Number of qubits", ylabel, title)
        _save(fig, os.path.join(out_dir, name))


def plot_entanglement(df, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(df["entanglement"], df["f1"], width=0.5,
                  color=ENCODING_COLORS["angle"])
    ax.bar_label(bars, fmt="%.3f", color=INK, padding=3)
    lo = max(0.0, df["f1"].min() - 0.05)
    ax.set_ylim(lo, 1.0)
    _style(ax, "Entanglement pattern", f"Test F1 (axis from {lo:.2f})",
           "F1 vs. entanglement pattern")
    _save(fig, os.path.join(out_dir, "ablation_entanglement.png"))


def write_summary(sweep_dfs, all_df, path):
    cols = list(KEY_FIELDS) + ["val_f1", "accuracy", "recall", "f1", "roc_auc",
                               "trainable_params", "train_time_sec"]
    lines = ["# Ablation summary", ""]
    for name, df in sweep_dfs.items():
        lines += [f"## Sweep: {name}", "", df[cols].to_markdown(index=False, floatfmt=".4f"), ""]
    best = all_df.sort_values("val_f1", ascending=False).iloc[0]
    lines += [
        "## Best configuration (chosen on validation F1)", "",
        ", ".join(f"{k}={best[k]}" for k in KEY_FIELDS), "",
        f"val F1 {best['val_f1']:.4f} -> test accuracy {best['accuracy']:.4f}, "
        f"test F1 {best['f1']:.4f}, test ROC-AUC {best['roc_auc']:.4f}", "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved {path}")


def run(subset: bool, dataset: str, sweeps) -> None:
    base_cfg = get_config(subset=subset, dataset=dataset)
    out_dir = base_cfg.results_dir
    os.makedirs(out_dir, exist_ok=True)

    cache = {}  # config key -> result row, so shared configs train once
    sweep_dfs = {}
    for name in sweeps:
        rows = []
        for overrides in SWEEPS[name]:
            cfg = replace(base_cfg, **overrides)
            key = tuple(getattr(cfg, k) for k in KEY_FIELDS)
            if key not in cache:
                print(f"\n=== {name}: " + " ".join(f"{k}={v}" for k, v in zip(KEY_FIELDS, key)) + " ===")
                cache[key] = run_one(cfg, subset)
                r = cache[key]
                print(f"-> acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
                      f"val_f1={r['val_f1']:.4f} time={r['train_time_sec']:.1f}s")
            rows.append(cache[key])
        df = pd.DataFrame(rows)
        sweep_dfs[name] = df
        df.to_csv(os.path.join(out_dir, f"ablation_{name}.csv"), index=False)

    all_df = pd.DataFrame(list(cache.values()))
    all_df.to_csv(os.path.join(out_dir, "ablation_all.csv"), index=False)

    if "depth" in sweep_dfs:
        plot_depth(sweep_dfs["depth"], out_dir)
    if "qubits" in sweep_dfs:
        plot_qubits(sweep_dfs["qubits"], out_dir)
    if "entanglement" in sweep_dfs:
        plot_entanglement(sweep_dfs["entanglement"], out_dir)
    write_summary(sweep_dfs, all_df, os.path.join(out_dir, "ablation_summary.md"))


def main():
    parser = argparse.ArgumentParser(description="Hybrid model circuit ablation (stage 3).")
    parser.add_argument("--subset", action="store_true",
                         help="Run on a small subset for fast debugging.")
    parser.add_argument("--dataset", choices=list(DATASET_PRESETS), default="tb",
                         help="Which dataset to run on (default: tb).")
    parser.add_argument("--sweeps", nargs="+", choices=list(SWEEPS), default=list(SWEEPS),
                         help="Which sweeps to run (default: all).")
    args = parser.parse_args()
    run(args.subset, args.dataset, args.sweeps)


if __name__ == "__main__":
    main()
