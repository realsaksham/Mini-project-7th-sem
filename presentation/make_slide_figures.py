"""
Projector-friendly figures for the slide deck (large fonts, honest axes).

    python presentation/make_slide_figures.py

Reads only saved results (results/tb/...) and the dataset; no retraining.
Writes PNGs to presentation/figures/.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from configs.config import get_config
from src.data import _IMAGE_EXTS, _TB_CLASS_DIRS

RES = "results/tb"
OUT = "presentation/figures"
os.makedirs(OUT, exist_ok=True)

INK = "#1B1B1F"
MUTED = "#55555E"
BLUE = "#2A78D6"      # classical
ORANGE = "#EB6834"    # quantum
GREY = "#9AA0A8"      # controls
ONE_IMAGE = 100 / 630  # one test image, in accuracy percentage points

plt.rcParams.update({
    "font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Dataset: file paths in the same order as the cached arrays, and the test split
# ---------------------------------------------------------------------------
cfg = get_config()
paths, labels = [], []
for class_dir, label in _TB_CLASS_DIRS.items():
    folder = os.path.join(cfg.tb_root, class_dir)
    for n in sorted(os.listdir(folder)):
        if n.lower().endswith(_IMAGE_EXTS):
            paths.append(os.path.join(folder, n))
            labels.append(label)
labels = np.asarray(labels)
idx = np.arange(len(labels))
_, temp_idx = train_test_split(idx, test_size=cfg.val_frac + cfg.test_frac,
                               stratify=labels, random_state=cfg.split_seed)
_, test_idx = train_test_split(temp_idx, test_size=cfg.test_frac / (cfg.val_frac + cfg.test_frac),
                               stratify=labels[temp_idx], random_state=cfg.split_seed)


def load_xray(i, size=320):
    with Image.open(paths[i]) as im:
        return np.asarray(im.convert("L").resize((size, size), Image.BILINEAR))


# ---------------------------------------------------------------------------
# 1. Sample X-rays with large class labels (no file names)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 3, figsize=(9, 6.4))
for row, (label, name, color) in enumerate([(0, "Normal", BLUE), (1, "TB", ORANGE)]):
    pick = rng.choice(np.flatnonzero(labels == label), 3, replace=False)
    for ax, i in zip(axes[row], pick):
        ax.imshow(load_xray(i), cmap="gray")
        ax.axis("off")
    axes[row, 0].text(-0.08, 0.5, name, transform=axes[row, 0].transAxes, rotation=90,
                      va="center", ha="right", fontsize=26, fontweight="bold", color=color)
fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.08)
save(fig, "samples_labeled.png")

# ---------------------------------------------------------------------------
# 2. Ablation: one bar chart, all panels on the same 96-99% axis
# ---------------------------------------------------------------------------
ab = pd.read_csv(f"{RES}/ablation_all.csv")
ab["acc"] = 100 * ab.accuracy
ang = ab[ab.encoding == "angle"]
panels = [
    ("Depth (4 qubits)", [(str(d), ang[(ang.n_qubits == 4) & (ang.entanglement == "strong") &
                                       (ang.circuit_depth == d)].acc.iloc[0], ORANGE) for d in (1, 2, 3, 4)]),
    ("Qubits (depth 2)", [(str(q), ang[(ang.n_qubits == q) & (ang.entanglement == "strong") &
                                       (ang.circuit_depth == 2)].acc.iloc[0], ORANGE) for q in (2, 4, 6, 8)]),
    ("Entanglement", [(e, ang[(ang.n_qubits == 4) & (ang.circuit_depth == 2) &
                              (ang.entanglement == e)].acc.iloc[0], ORANGE) for e in ("strong", "basic", "none")]),
    ("Amplitude encoding", [(str(q), ab[(ab.encoding == "amplitude") & (ab.n_qubits == q)].acc.iloc[0], GREY)
                            for q in (2, 4, 6, 8)]),
]
fig, axes = plt.subplots(1, 4, figsize=(16, 5.2), sharey=True,
                         gridspec_kw={"width_ratios": [4, 4, 3, 4]})
for ax, (title, bars) in zip(axes, panels):
    xs = np.arange(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.65)
    for x, b in zip(xs, bars):
        ax.text(x, b[1] + 0.06, f"{b[1]:.1f}", ha="center", va="bottom", fontsize=15)
    ax.set_xticks(xs, [b[0] for b in bars])
    ax.set_title(title, fontsize=19, pad=10)
    ax.set_ylim(96, 99.5)
axes[0].set_ylabel("Test accuracy (%)")
axes[0].set_xlabel("layers"); axes[1].set_xlabel("qubits"); axes[3].set_xlabel("qubits")
fig.text(0.5, -0.06, f"Same axis on every panel.  One test image = {ONE_IMAGE:.2f} percentage points.",
         ha="center", fontsize=17, color=MUTED)
fig.tight_layout()
save(fig, "ablation_bars.png")

# ---------------------------------------------------------------------------
# 3. Backbone x head
# ---------------------------------------------------------------------------
comp = pd.read_csv(f"{RES}/comparison/summary.csv", index_col=0)
fig, ax = plt.subplots(figsize=(8, 5.6))
for j, (backbone, label) in enumerate([("CNN", "Small CNN\n(132 k params)"),
                                       ("ResNet18", "ResNet18, pretrained\n(11.2 M params)")]):
    for k, (head, color) in enumerate([("classical", BLUE), ("quantum", ORANGE)]):
        r = comp.loc[f"{backbone} + {head}"]
        x = j + (k - 0.5) * 0.36
        ax.bar(x, 100 * r.accuracy_mean, 0.34, yerr=100 * r.accuracy_std, capsize=6, color=color,
               label=f"{head} head" if j == 0 else None)
        ax.text(x, 100 * (r.accuracy_mean + r.accuracy_std) + 0.08, f"{100 * r.accuracy_mean:.2f}",
                ha="center", va="bottom", fontsize=16)
ax.set_xticks([0, 1], ["Small CNN", "ResNet18 (pretrained)"])
ax.set_ylim(96, 100.3)
ax.set_ylabel("Test accuracy (%)")
ax.legend(loc="upper left", frameon=False)
ax.set_title("Mean ± std over 3 seeds", fontsize=17, color=MUTED)
fig.tight_layout()
save(fig, "backbone_head.png")

# ---------------------------------------------------------------------------
# 4. Main study: accuracy of the seven models (100% data)
# ---------------------------------------------------------------------------
study = pd.read_csv(f"{RES}/hybrid_study/summary.csv")
full = study[study.train_fraction == 1.0].set_index("model")
models = [("Classical CNN", "Classical\nCNN", BLUE),
          ("CNN + matched classical head", "CNN +\nclassical twin", GREY),
          ("C->Q hybrid (CNN + quantum head)", "C→Q\nhybrid", ORANGE),
          ("Q->C hybrid (quanvolution + CNN)", "Q→C\nhybrid", ORANGE),
          ("Random patch + CNN (control)", "Random\nfilter", GREY),
          ("Avg-pool + CNN (control)", "Average\npooling", GREY),
          ("Learned 2x2 conv + CNN (control)", "Learned\nfilter", GREY)]
fig, ax = plt.subplots(figsize=(13, 5.6))
for x, (key, label, color) in enumerate(models):
    r = full.loc[key]
    m, s = 100 * r.accuracy_mean, 100 * r.accuracy_std
    ax.bar(x, m, 0.62, yerr=s, capsize=6, color=color)
    ax.text(x, m + s + 0.12, f"{m:.2f}", ha="center", va="bottom", fontsize=16)
ax.set_xticks(range(len(models)), [m[1] for m in models], fontsize=15)
ax.set_ylim(89, 100)
ax.set_ylabel("Test accuracy (%)")
ax.axvline(2.5, color=MUTED, lw=1, ls=":")
ax.text(1, 99.4, "Testing C→Q", ha="center", fontsize=16, color=MUTED)
ax.text(4.5, 99.4, "Testing Q→C", ha="center", fontsize=16, color=MUTED)
fig.tight_layout()
save(fig, "main_study_bars.png")

# ---------------------------------------------------------------------------
# 5. Low-data F1: four lines, labelled directly
# ---------------------------------------------------------------------------
lines = [("Classical CNN", "Classical CNN", BLUE, "-"),
         ("C->Q hybrid (CNN + quantum head)", "C→Q hybrid", ORANGE, "-"),
         ("Q->C hybrid (quanvolution + CNN)", "Q→C hybrid", "#B03A2E", "-"),
         ("Avg-pool + CNN (control)", "Average pooling + CNN", GREY, "--")]
fig, ax = plt.subplots(figsize=(9.5, 5.6))
for key, label, color, ls in lines:
    d = study[study.model == key].sort_values("train_fraction")
    x = 100 * d.train_fraction.values
    ax.errorbar(x, d.f1_mean, yerr=d.f1_std, color=color, ls=ls, lw=3, marker="o", ms=8, capsize=4,
                label=label)
ax.legend(loc="lower right", frameon=False, fontsize=18)
ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 25, 100], ["2%\n(58)", "5%", "10%", "25%", "100%\n(2,940)"])
ax.set_xlim(1.6, 130)
ax.set_ylim(0, 1.0)
ax.set_xlabel("Share of training data (images)")
ax.set_ylabel("Test F1 (mean ± std)")
fig.tight_layout()
save(fig, "lowdata_f1.png")

# ---------------------------------------------------------------------------
# 6. NISQ robustness: shots and depolarising noise on a wide axis
# ---------------------------------------------------------------------------
hw = pd.read_csv(f"{RES}/quantum_analysis/hardware_realism.csv")
exact = 100 * hw[hw.setting == "exact"].accuracy.iloc[0]
shots = hw[hw.setting == "shots"]
noise = hw[hw.setting == "depolarizing"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, x, y, xlabel in [
        (axes[0], shots.shots.values, 100 * shots.accuracy.values, "Measurement shots per circuit"),
        (axes[1], noise.noise_p.values, 100 * noise.accuracy.values, "Depolarising noise p per layer")]:
    ax.axhspan(exact - ONE_IMAGE, exact + ONE_IMAGE, color=BLUE, alpha=0.15, lw=0)
    ax.axhline(exact, color=BLUE, lw=1.5, ls="--")
    ax.plot(x, y, color=ORANGE, lw=3, marker="o", ms=9)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylim(90, 100)
axes[0].set_ylabel("Test accuracy (%)")
axes[0].text(12, exact - 1.2, f"noiseless: {exact:.2f}%\nshaded band = ±1 test image",
             color=BLUE, fontsize=16, va="top")
fig.tight_layout()
save(fig, "nisq_robustness.png")

# ---------------------------------------------------------------------------
# 7. Error analysis: CNN + quantum head (seed 42), large labels
# ---------------------------------------------------------------------------
mis = pd.read_csv(f"{RES}/comparison/misclassified.csv")
names = {0: "Normal", 1: "Tuberculosis"}
# Self-check: the reconstructed test order must reproduce the stored labels.
assert all(names[labels[test_idx[r.test_index]]] == r.true_label for r in mis.itertuples()), \
    "test split reconstruction does not match misclassified.csv"
err = mis[mis.wrong_cnn_quantum].copy()
err["conf"] = (err.p_cnn_quantum - 0.5).abs()
err = err.sort_values("conf", ascending=False).head(6)
fig, axes = plt.subplots(1, len(err), figsize=(3.0 * len(err), 4.0))
for ax, r in zip(np.atleast_1d(axes), err.itertuples()):
    ax.imshow(load_xray(test_idx[r.test_index]), cmap="gray")
    ax.axis("off")
    is_tb = r.true_label == "Tuberculosis"
    ax.set_title(f"{'TB' if is_tb else 'Normal'}\nP(TB) = {r.p_cnn_quantum:.2f}", fontsize=20,
                 color=ORANGE if is_tb else BLUE, fontweight="bold")
fig.tight_layout()
save(fig, "errors_labeled.png")

# ---------------------------------------------------------------------------
# 8. QR code for the code repository
# ---------------------------------------------------------------------------
import qrcode
qrcode.make("https://github.com/realsaksham/Mini-project-7th-sem", border=2).save(f"{OUT}/repo_qr.png")
print("wrote repo_qr.png")
