"""
Stage-4/5: the 4-way comparison + final plots.

    {custom CNN, pretrained ResNet18}  x  {classical head, quantum head}

Each of the 4 models is trained with several seeds (the train/val/test split
is fixed by cfg.split_seed, so seeds only change weight init + batch order)
and reported as mean +- std, so a difference between models can be judged
against run-to-run noise rather than read off a single run.

Outputs (in <results_dir>/comparison/):
    runs.csv                 every (model, seed) run's val + test metrics
    summary.csv / summary.md mean +- std per model
    roc_curves.png           ROC of all 4 models (first seed)
    confusion_<model>.png    confusion matrix per model (first seed)
    misclassified.csv        every test image any model got wrong (first seed)
    errors_<model>.png       grid of that model's misclassified test X-rays

Usage:
    python -m src.compare --subset           # fast debug run
    python -m src.compare                    # full run, seeds 42 43 44
    python -m src.compare --seeds 42         # single seed
"""

import argparse
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configs.config import DATASET_PRESETS, get_config
from src.data import load_data, set_seed
from src.evaluate import (compute_metrics, predict_probs, save_confusion_matrix,
                          save_roc_curves)
from src.models import build_model
from src.train import count_trainable_params, train_model

MODELS = [
    # (name, backbone_name, head_type)
    ("CNN + classical", "small_cnn", "classical"),
    ("CNN + quantum", "small_cnn", "quantum"),
    ("ResNet18 + classical", "resnet18", "classical"),
    ("ResNet18 + quantum", "resnet18", "quantum"),
]
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")

# Categorical slots 1-4 of the dataviz skill's validated default palette, in
# fixed order; line style adds a second, non-colour cue (solid = CNN,
# dashed = ResNet18).
STYLES = {
    "CNN + classical": dict(color="#2a78d6", ls="-"),
    "CNN + quantum": dict(color="#eb6834", ls="-"),
    "ResNet18 + classical": dict(color="#1baf7a", ls="--"),
    "ResNet18 + quantum": dict(color="#eda100", ls="--"),
}


def slug(name: str) -> str:
    return name.lower().replace(" + ", "_").replace(" ", "_")


def save_error_grid(images, labels, probs, class_names, path, title, max_images=24):
    """Grid of misclassified test X-rays, each titled with truth + P(positive)."""
    wrong = np.flatnonzero((probs >= 0.5).astype(int) != labels)
    # Most confident mistakes first -- those are the most informative.
    wrong = wrong[np.argsort(-np.abs(probs[wrong] - 0.5))][:max_images]
    if len(wrong) == 0:
        return
    cols = 6
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.95))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"true: {class_names[int(labels[i])]}\nP({class_names[1][:2]})={probs[i]:.2f}",
                     fontsize=7, color="#0b0b0b")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def test_images_uint8(test_loader):
    """Recover display-ready uint8 images (in loader order) from the test loader."""
    batches = [((x.squeeze(1).numpy() * 0.5 + 0.5) * 255).round().astype(np.uint8)
               for x, _ in test_loader]
    return np.concatenate(batches)


def run(subset: bool, dataset: str, seeds) -> None:
    base_cfg = get_config(subset=subset, dataset=dataset)
    out_dir = os.path.join(base_cfg.results_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    first_seed = {}  # model name -> (labels, probs) on the test set, for plots
    test_images = None
    for name, backbone, head in MODELS:
        for seed in seeds:
            cfg = replace(base_cfg, backbone_name=backbone, seed=seed)
            set_seed(seed)
            print(f"\n=== {name} | seed {seed} ===")
            train_loader, val_loader, test_loader = load_data(cfg, subset=subset)
            model = build_model(cfg, head_type=head)
            n_params = count_trainable_params(model)

            _, duration_sec = train_model(model, train_loader, val_loader, cfg)
            val_probs, val_labels = predict_probs(model, val_loader)
            test_probs, test_labels = predict_probs(model, test_loader)
            val_m, _, _ = compute_metrics(val_probs, val_labels)
            test_m, _, _ = compute_metrics(test_probs, test_labels)

            row = {"model": name, "backbone": backbone, "head": head, "seed": seed}
            row.update({f"val_{m}": val_m[m] for m in METRICS})
            row.update({m: test_m[m] for m in METRICS})
            row.update(trainable_params=n_params, train_time_sec=duration_sec)
            rows.append(row)
            print(f"-> test acc={test_m['accuracy']:.4f} f1={test_m['f1']:.4f} "
                  f"auc={test_m['roc_auc']:.4f} time={duration_sec:.1f}s")

            if seed == seeds[0]:
                first_seed[name] = (test_labels, test_probs)
                if test_images is None:
                    test_images = test_images_uint8(test_loader)

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)

    # --- mean +- std per model ---
    agg_cols = [f"val_{m}" for m in METRICS] + list(METRICS) + ["train_time_sec"]
    order = [m[0] for m in MODELS]
    grouped = runs.groupby("model", sort=False)
    summary = grouped[agg_cols].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary["trainable_params"] = grouped["trainable_params"].first()
    summary = summary.loc[order]
    summary.to_csv(os.path.join(out_dir, "summary.csv"))

    def pm(r, c):
        std = r[f"{c}_std"]
        return f"{r[f'{c}_mean']:.4f} ± {std:.4f}" if not np.isnan(std) else f"{r[f'{c}_mean']:.4f}"

    md = ["# 4-way comparison", "",
          f"Seeds: {', '.join(map(str, seeds))} (mean ± std over seeds; fixed data split). "
          "Test set metrics unless noted.", "",
          "| model | accuracy | precision | recall | F1 | ROC-AUC | val F1 | params | train time (s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, r in summary.iterrows():
        md.append(f"| {name} | {pm(r, 'accuracy')} | {pm(r, 'precision')} | {pm(r, 'recall')} | "
                  f"{pm(r, 'f1')} | {pm(r, 'roc_auc')} | {pm(r, 'val_f1')} | "
                  f"{int(r['trainable_params']):,} | {r['train_time_sec_mean']:.0f} |")

    # --- error analysis (first seed) ---
    labels = next(iter(first_seed.values()))[0]
    wrong = {name: (probs >= 0.5).astype(int) != labels for name, (_, probs) in first_seed.items()}
    any_wrong = np.flatnonzero(np.any(list(wrong.values()), axis=0))
    mis = pd.DataFrame({"test_index": any_wrong,
                        "true_label": [base_cfg.class_names[int(labels[i])] for i in any_wrong]})
    for name, (_, probs) in first_seed.items():
        mis[f"p_{slug(name)}"] = probs[any_wrong].round(4)
        mis[f"wrong_{slug(name)}"] = wrong[name][any_wrong]
    mis["n_models_wrong"] = mis[[c for c in mis if c.startswith("wrong_")]].sum(axis=1)
    mis.to_csv(os.path.join(out_dir, "misclassified.csv"), index=False)

    md += ["", f"## Error analysis (seed {seeds[0]}, {len(labels)} test images)", "",
           "| model | errors | missed TB (false negatives) | false alarms (false positives) |",
           "|---|---|---|---|"]
    for name, w in wrong.items():
        fn = int(np.sum(w & (labels == 1)))
        fp = int(np.sum(w & (labels == 0)))
        md.append(f"| {name} | {int(w.sum())} | {fn} | {fp} |")
    all_wrong = int(np.sum(np.all(list(wrong.values()), axis=0)))
    md += ["", f"Images misclassified by at least one model: {len(any_wrong)}; "
           f"by all four models: {all_wrong}.", ""]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n" + "\n".join(md))

    # --- plots (first seed) ---
    save_roc_curves(
        {name: (lab, probs, STYLES[name]) for name, (lab, probs) in first_seed.items()},
        os.path.join(out_dir, "roc_curves.png"),
        title=f"ROC curves, test set (seed {seeds[0]})",
    )
    for name, (lab, probs) in first_seed.items():
        save_confusion_matrix(lab, (probs >= 0.5).astype(int),
                              os.path.join(out_dir, f"confusion_{slug(name)}.png"),
                              title=name, class_names=base_cfg.class_names)
        save_error_grid(test_images, lab, probs, base_cfg.class_names,
                        os.path.join(out_dir, f"errors_{slug(name)}.png"),
                        title=f"{name}: misclassified test images (most confident first)")
    print(f"\nSaved outputs to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="4-way backbone x head comparison (stages 4-5).")
    parser.add_argument("--subset", action="store_true",
                         help="Run on a small subset for fast debugging.")
    parser.add_argument("--dataset", choices=list(DATASET_PRESETS), default="tb",
                         help="Which dataset to run on (default: tb).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                         help="Training seeds; results are reported as mean +- std.")
    args = parser.parse_args()
    run(args.subset, args.dataset, args.seeds)


if __name__ == "__main__":
    main()
