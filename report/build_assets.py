"""
Build every figure and table the report needs, straight from results/ and the
code, so no number in the report is typed by hand.

    python report/build_assets.py

Writes report/figures/*.png and report/tables/*.tex.
"""

import json
import os
import shutil
import sys
from dataclasses import asdict, replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from PIL import Image

from configs.config import Config, get_config
from src.data import load_data
from src.models import make_quantum_circuit
from src.quanvolution import avgpool_features, make_quanv_circuit, quanv_features
from src.hybrid_study import MODELS as STUDY_MODELS, PAIRS, mcnemar_p

RES = "results/tb"
FIG = "report/figures"
TAB = "report/tables"
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

INK, MUTED = "#0b0b0b", "#52514e"


def tex_escape(s: str) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%")
            .replace("_", "\\_").replace("#", "\\#").replace("->", "$\\rightarrow$")
            .replace("±", "$\\pm$").replace("×", "$\\times$"))


def write_table(name, caption, label, header, rows, align=None, note=None, size="\\small"):
    align = align or "l" + "r" * (len(header) - 1)
    lines = ["\\begin{table}[htbp]", "\\centering", size,
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             # Shrinks (never enlarges) a table that is wider than the text block.
             "\\begin{adjustbox}{max width=\\linewidth}",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MIDRULE":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if note:
        lines += ["\\par\\vspace{2pt}", f"\\parbox{{0.92\\linewidth}}{{\\footnotesize {note}}}"]
    lines.append("\\end{table}")
    with open(os.path.join(TAB, name + ".tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("table", name)


def pm(mean, std, pct=False, digits=4):
    if pct:
        return f"{100 * mean:.2f} $\\pm$ {100 * std:.2f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def pct(x):
    return f"{100 * x:.2f}"


# ---------------------------------------------------------------------------
# Copy result figures
# ---------------------------------------------------------------------------

COPY = {
    "confusion_matrix_baseline.png": f"{RES}/confusion_matrix_baseline.png",
    "confusion_matrix_hybrid.png": f"{RES}/confusion_matrix_hybrid.png",
    "ablation_depth.png": f"{RES}/ablation_depth.png",
    "ablation_qubits.png": f"{RES}/ablation_qubits.png",
    "ablation_time_qubits.png": f"{RES}/ablation_time_qubits.png",
    "ablation_entanglement.png": f"{RES}/ablation_entanglement.png",
    "roc_curves.png": f"{RES}/comparison/roc_curves.png",
    "errors_cnn_quantum.png": f"{RES}/comparison/errors_cnn_quantum.png",
    "errors_cnn_classical.png": f"{RES}/comparison/errors_cnn_classical.png",
    "confusion_resnet18_quantum.png": f"{RES}/comparison/confusion_resnet18_quantum.png",
    "confusion_resnet18_classical.png": f"{RES}/comparison/confusion_resnet18_classical.png",
    "f1_vs_data.png": f"{RES}/hybrid_study/f1_vs_data.png",
    "auc_vs_data.png": f"{RES}/hybrid_study/auc_vs_data.png",
    "expressibility.png": f"{RES}/quantum_analysis/expressibility.png",
    "entangling_capability.png": f"{RES}/quantum_analysis/entangling_capability.png",
    "barren_plateaus.png": f"{RES}/quantum_analysis/barren_plateaus.png",
    "noise_robustness.png": f"{RES}/quantum_analysis/noise_robustness.png",
    "shots_robustness.png": f"{RES}/quantum_analysis/shots_robustness.png",
}
for dst, src in COPY.items():
    shutil.copy(src, os.path.join(FIG, dst))
print("copied", len(COPY), "result figures")


# ---------------------------------------------------------------------------
# New figures
# ---------------------------------------------------------------------------

cfg = get_config()

# Sample images (original resolution)
fig, axes = plt.subplots(2, 4, figsize=(8, 4.3))
for row, cls in enumerate(["Normal", "Tuberculosis"]):
    folder = os.path.join(cfg.tb_root, cls)
    names = sorted(os.listdir(folder))[:: max(1, len(os.listdir(folder)) // 4)][:4]
    for ax, n in zip(axes[row], names):
        ax.imshow(Image.open(os.path.join(folder, n)).convert("L"), cmap="gray")
        ax.set_title(f"{cls}\n{n}", fontsize=7, color=INK)
        ax.axis("off")
fig.tight_layout()
fig.savefig(f"{FIG}/sample_images.png", dpi=150)
plt.close(fig)

# Quanvolution feature maps for one TB image
images = np.load(f"{cfg.data_root}/tb_chest_xray_64.npz")
img_arr, lab_arr = images["images"], images["labels"]
tb_idx = int(np.flatnonzero(lab_arr == 1)[0])
one = img_arr[tb_idx:tb_idx + 1]
q = quanv_features(one, cfg.quanv_layers, cfg.quanv_seed)[0]
a = avgpool_features(one)[0, 0]
fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
axes[0].imshow(one[0], cmap="gray"); axes[0].set_title("Input 64×64", fontsize=9)
axes[1].imshow(a, cmap="gray"); axes[1].set_title("2×2 avg-pool 32×32", fontsize=9)
for c in range(4):
    axes[2 + c].imshow(q[c], cmap="RdBu", vmin=-1, vmax=1)
    axes[2 + c].set_title(f"Quanv channel {c} ⟨Z{c}⟩", fontsize=9)
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(f"{FIG}/quanv_feature_maps.png", dpi=150)
plt.close(fig)

# Circuit diagrams
qml.drawer.use_style("black_white")
for ent, name in [("strong", "circuit_cq_strong"), ("basic", "circuit_cq_basic"), ("none", "circuit_cq_none")]:
    circuit, shapes = make_quantum_circuit(4, 2, "angle", ent)
    fig, _ = qml.draw_mpl(circuit, level="device", decimals=None)(
        torch.tensor([0.1, 0.2, 0.3, 0.4]), torch.zeros(shapes["weights"]))
    fig.savefig(f"{FIG}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
qc, w = make_quanv_circuit(cfg.quanv_layers, cfg.quanv_seed)
fig, _ = qml.draw_mpl(qc, level="device", decimals=2)(torch.full((4,), 0.5), w)
fig.savefig(f"{FIG}/circuit_quanv.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("circuit diagrams drawn")


# Architecture block diagrams
def blocks(ax, y, items, colors):
    x = 0.01
    width = (0.98 - 0.025 * (len(items) - 1)) / len(items)
    for i, (txt, col) in enumerate(zip(items, colors)):
        ax.add_patch(FancyBboxPatch((x, y), width, 0.16, boxstyle="round,pad=0.005,rounding_size=0.015",
                                    fc=col, ec="#52514e", lw=1))
        ax.text(x + width / 2, y + 0.08, txt, ha="center", va="center", fontsize=8.2, color=INK, wrap=True)
        if i < len(items) - 1:
            ax.annotate("", xy=(x + width + 0.025, y + 0.08), xytext=(x + width, y + 0.08),
                        arrowprops=dict(arrowstyle="->", color="#52514e", lw=1.2))
        x += width + 0.025


C_CL, C_Q, C_IO = "#dbe9fa", "#fde3d6", "#eeeeea"
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
rows = [
    ("(a) Classical CNN baseline", ["X-ray\n64×64×1", "Conv 3×3 (8)\nReLU, MaxPool", "Conv 3×3 (16)\nReLU, MaxPool",
                                    "Dense\n4096→32", "Linear 32→1\nsigmoid", "P(TB)"],
     [C_IO, C_CL, C_CL, C_CL, C_CL, C_IO]),
    ("(b) C→Q hybrid: classical features → quantum classifier",
     ["X-ray\n64×64×1", "CNN backbone\n(as in a)", "Linear 32→4\ntanh·π", "4-qubit VQC\nRY enc. + SEL ×2",
      "⟨Z⟩ ×4 →\nLinear 4→1", "P(TB)"],
     [C_IO, C_CL, C_CL, C_Q, C_CL, C_IO]),
    ("(c) Q→C hybrid: quantum convolution → classical CNN",
     ["X-ray\n64×64×1", "2×2 patches\nstride 2", "4-qubit quanv\nfilter (frozen)", "Quantum map\n32×32×4",
      "CNN backbone\n+ Linear 32→1", "P(TB)"],
     [C_IO, C_CL, C_Q, C_Q, C_CL, C_IO]),
]
for i, (title, items, cols) in enumerate(rows):
    y = 0.70 - i * 0.33
    ax.text(0.01, y + 0.2, title, fontsize=10, color=INK, weight="bold")
    blocks(ax, y, items, cols)
ax.text(0.99, 0.0, "blue = classical,  orange = quantum (simulated)", ha="right", fontsize=8, color=MUTED)
fig.savefig(f"{FIG}/architectures.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# Overall workflow
fig, ax = plt.subplots(figsize=(11, 1.9))
ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 0.3)
blocks(ax, 0.05, ["Kaggle TB CXR\n4,200 images", "Grayscale, 64×64\nnormalise", "Stratified split\n70/15/15",
                  "Train 7+4 models\n3 seeds, 5 data sizes", "Metrics, McNemar\nROC, errors",
                  "Circuit analysis\nexpr., BP, noise"],
       [C_IO, C_CL, C_CL, C_Q, C_CL, C_Q])
fig.savefig(f"{FIG}/workflow.png", dpi=170, bbox_inches="tight")
plt.close(fig)
print("diagrams drawn")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

# Baseline vs hybrid (single seed 42)
b = json.load(open(f"{RES}/baseline.json"))
h = json.load(open(f"{RES}/hybrid.json"))
rows = []
for key, lab in [("accuracy", "Accuracy (\\%)"), ("precision", "Precision"), ("recall", "Recall"),
                 ("f1", "F1-score"), ("roc_auc", "ROC-AUC")]:
    if key == "accuracy":
        rows.append([lab, pct(b[key]), pct(h[key])])
    else:
        rows.append([lab, f"{b[key]:.4f}", f"{h[key]:.4f}"])
rows.append(["Trainable parameters", f"{b['trainable_params']:,}", f"{h['trainable_params']:,}"])
rows.append(["Training time (s)", f"{b['train_time_sec']:.1f}", f"{h['train_time_sec']:.1f}"])
write_table("baseline_vs_hybrid", "Single-run comparison of the classical baseline and the C$\\rightarrow$Q hybrid "
            "(seed 42, full training set, 630 test images).", "tab:baseline_vs_hybrid",
            ["Metric", "Classical CNN", "C$\\rightarrow$Q hybrid"], rows)

# Ablation
ab = pd.read_csv(f"{RES}/ablation_all.csv")
for sweep, title, label in [("depth", "circuit depth", "tab:abl_depth"),
                            ("entanglement", "entanglement pattern", "tab:abl_ent"),
                            ("qubits", "number of qubits and encoding", "tab:abl_qubits")]:
    df = pd.read_csv(f"{RES}/ablation_{sweep}.csv")
    rows = [[str(r.n_qubits), str(r.circuit_depth), r.encoding, r.entanglement, f"{r.val_f1:.4f}",
             pct(r.accuracy), f"{r.recall:.4f}", f"{r.f1:.4f}", f"{r.roc_auc:.4f}",
             f"{r.trainable_params:,}", f"{r.train_time_sec:.0f}"] for r in df.itertuples()]
    write_table(f"ablation_{sweep}", f"Circuit ablation: effect of the {title} on the C$\\rightarrow$Q hybrid "
                "(seed 42, test set unless marked val).", label,
                ["Qubits", "Depth", "Encoding", "Entangl.", "Val F1", "Acc. (\\%)", "Recall", "F1", "AUC",
                 "Params", "Time (s)"], rows, align="rrllrrrrrrr", size="\\footnotesize")

# 4-way comparison
cs = pd.read_csv(f"{RES}/comparison/summary.csv", index_col=0)
rows = [[tex_escape(m), f"{int(r.trainable_params):,}", pm(r.accuracy_mean, r.accuracy_std, pct=True),
         pm(r.precision_mean, r.precision_std), pm(r.recall_mean, r.recall_std), pm(r.f1_mean, r.f1_std),
         pm(r.roc_auc_mean, r.roc_auc_std), f"{r.train_time_sec_mean:.0f}"] for m, r in cs.iterrows()]
write_table("comparison", "Backbone $\\times$ head comparison: mean $\\pm$ standard deviation over seeds "
            "42, 43 and 44 on the fixed test set.", "tab:comparison",
            ["Model", "Params", "Accuracy (\\%)", "Precision", "Recall", "F1", "ROC-AUC", "Time (s)"],
            rows, size="\\footnotesize")

mis = pd.read_csv(f"{RES}/comparison/misclassified.csv")
lab_test = (mis["true_label"] == "Tuberculosis").to_numpy()
rows = []
for m in cs.index:
    slug = m.lower().replace(" + ", "_").replace(" ", "_")
    w = mis[f"wrong_{slug}"].to_numpy()
    rows.append([tex_escape(m), str(int(w.sum())), str(int((w & lab_test).sum())), str(int((w & ~lab_test).sum()))])
write_table("comparison_errors", "Error breakdown on the 630 test images (seed 42). Images misclassified by at "
            f"least one model: {len(mis)}; by all four models: {int((mis['n_models_wrong'] == 4).sum())}.",
            "tab:comparison_errors", ["Model", "Errors", "Missed TB (FN)", "False alarms (FP)"], rows)

# Hybrid study
hs = pd.read_csv(f"{RES}/hybrid_study/summary.csv")
order = [m[0] for m in STUDY_MODELS]
full = hs[hs.train_fraction == 1.0].set_index("model").loc[order]
rows = [[tex_escape(m), f"{int(r.trainable_params):,}", str(int(r.head_params)),
         pm(r.accuracy_mean, r.accuracy_std, pct=True), pm(r.recall_mean, r.recall_std),
         pm(r.f1_mean, r.f1_std), pm(r.roc_auc_mean, r.roc_auc_std), f"{r.train_time_sec_mean:.0f}"]
        for m, r in full.iterrows()]
write_table("study_full", "Main study on the full training set (2,940 images): mean $\\pm$ std over three seeds. "
            "Head parameters count only the classifier head.", "tab:study_full",
            ["Model", "Params", "Head", "Accuracy (\\%)", "Recall", "F1", "ROC-AUC", "Time (s)"],
            rows, size="\\footnotesize")

fracs = sorted(hs.train_fraction.unique())
rows = []
for m in order:
    g = hs[hs.model == m].set_index("train_fraction")
    rows.append([tex_escape(m)] + [pm(g.loc[f, "f1_mean"], g.loc[f, "f1_std"], digits=3) for f in fracs])
runs = pd.read_csv(f"{RES}/hybrid_study/runs.csv")
ntrain = [str(int(runs[runs.train_fraction == f].n_train.iloc[0])) for f in fracs]
write_table("study_lowdata_f1", "Test F1 (mean $\\pm$ std, three seeds) versus the fraction of the training "
            "split used. Validation and test sets are never reduced.", "tab:study_lowdata",
            ["Model"] + [f"{100 * f:g}\\% ({n})" for f, n in zip(fracs, ntrain)], rows,
            size="\\scriptsize")

rows = []
for m in order:
    g = hs[hs.model == m].set_index("train_fraction")
    rows.append([tex_escape(m)] + [pm(g.loc[f, "roc_auc_mean"], g.loc[f, "roc_auc_std"], digits=3) for f in fracs])
write_table("study_lowdata_auc", "Test ROC-AUC (mean $\\pm$ std, three seeds) versus the fraction of the "
            "training split used.", "tab:study_lowdata_auc",
            ["Model"] + [f"{100 * f:g}\\%" for f in fracs], rows, size="\\scriptsize")

# McNemar from saved test probabilities
cfg_full = get_config()
_, _, test_loader = load_data(cfg_full)
test_labels = np.concatenate([y.squeeze(1).numpy() for _, y in test_loader])
probs = np.load(f"{RES}/hybrid_study/test_probs_full_data.npz")
seeds = sorted({int(k.split("|seed")[1]) for k in probs.files})
rows = []
for a, bname in PAIRS:
    ps, counts = [], []
    for s in seeds:
        wa = (probs[f"{a}|seed{s}"] >= 0.5).astype(int) != test_labels
        wb = (probs[f"{bname}|seed{s}"] >= 0.5).astype(int) != test_labels
        p = mcnemar_p(wa, wb)
        ps.append("$<$0.001" if p < 0.001 else f"{p:.3f}")
        counts.append(f"{int((~wa & wb).sum())}/{int((wa & ~wb).sum())}")
    rows.append([tex_escape(a), tex_escape(bname)] + [f"{p} ({c})" for p, c in zip(ps, counts)])
write_table("mcnemar", "McNemar exact test between each hybrid and a classical comparator on the full-data "
            "runs. Each cell gives the p-value and, in brackets, the number of test images only the hybrid "
            "got right / only the comparator got right.", "tab:mcnemar",
            ["Hybrid", "Compared with"] + [f"Seed {s}" for s in seeds], rows, align="llrrr",
            size="\\scriptsize")

# Quantum analysis
ex = pd.read_csv(f"{RES}/quantum_analysis/expressibility.csv")
rows = [[r.entanglement, str(r.depth), f"{r.expressibility_kl:.4f}", f"{r.entangling_capability:.4f}"]
        for r in ex.itertuples()]
write_table("expressibility", "Expressibility (KL divergence to the Haar fidelity distribution; lower is more "
            "expressive) and entangling capability (mean Meyer--Wallach $Q$) of the trainable block, 4 qubits, "
            "5,000 random parameter pairs per row.", "tab:expressibility",
            ["Entanglement", "Depth", "Expressibility (KL)", "Entangling capability $Q$"], rows, align="lrrr")

bp = pd.read_csv(f"{RES}/quantum_analysis/barren_plateaus.csv")
piv = bp.pivot_table(index="n_qubits", columns=["cost", "depth"], values="grad_variance")
cols = list(piv.columns)
rows = [[str(n)] + [f"{piv.loc[n, c]:.2e}" for c in cols] for n in piv.index]
write_table("barren", "Variance of $\\partial C/\\partial\\theta$ over 500 random initialisations of "
            "StronglyEntanglingLayers, for a local cost $\\langle Z_0\\rangle$ and a global cost "
            "$\\langle Z^{\\otimes n}\\rangle$.", "tab:barren",
            ["Qubits"] + [f"{c[0]}, $L={c[1]}$" for c in cols], rows)

hw = pd.read_csv(f"{RES}/quantum_analysis/hardware_realism.csv")
rows = []
for r in hw.itertuples():
    setting = ("Exact simulator" if r.setting == "exact" else
               f"{int(r.shots):,} shots" if r.setting == "shots" else f"Depolarising $p={r.noise_p:g}$")
    rows.append([setting, pct(r.accuracy), f"{r.recall:.4f}", f"{r.f1:.4f}", f"{r.roc_auc:.4f}"])
write_table("hardware", "Trained C$\\rightarrow$Q model (seed 42) evaluated with finite measurement shots and "
            "with depolarising noise after every circuit layer (density-matrix simulation), 630 test images.",
            "tab:hardware", ["Setting", "Accuracy (\\%)", "Recall", "F1", "ROC-AUC"], rows)

# Appendix: hyperparameters
c = asdict(Config())
skip = {"class_names"}
rows = [[tex_escape(k), tex_escape(v)] for k, v in c.items() if k not in skip]
write_table("hyperparameters", "All configuration values (\\texttt{configs/config.py}); experiments override "
            "only the swept field.", "tab:hyperparameters", ["Field", "Default value"], rows, align="ll",
            size="\\footnotesize")

# Appendix: per-seed runs
cr = pd.read_csv(f"{RES}/comparison/runs.csv")
rows = [[tex_escape(r.model), str(r.seed), pct(r.accuracy), f"{r.precision:.4f}", f"{r.recall:.4f}",
         f"{r.f1:.4f}", f"{r.roc_auc:.4f}", f"{r.val_f1:.4f}", f"{r.train_time_sec:.0f}"] for r in cr.itertuples()]
write_table("appendix_comparison_runs", "Per-seed results of the backbone $\\times$ head comparison.",
            "tab:app_comparison_runs", ["Model", "Seed", "Acc. (\\%)", "Prec.", "Recall", "F1", "AUC",
                                        "Val F1", "Time (s)"], rows, size="\\scriptsize")
fr = runs[runs.train_fraction == 1.0]
rows = [[tex_escape(r.model), str(r.seed), pct(r.accuracy), f"{r.precision:.4f}", f"{r.recall:.4f}",
         f"{r.f1:.4f}", f"{r.roc_auc:.4f}", f"{r.val_f1:.4f}", f"{r.train_time_sec:.0f}"] for r in fr.itertuples()]
write_table("appendix_study_runs", "Per-seed results of the main study on the full training set.",
            "tab:app_study_runs", ["Model", "Seed", "Acc. (\\%)", "Prec.", "Recall", "F1", "AUC",
                                   "Val F1", "Time (s)"], rows, size="\\scriptsize")

# Key numbers the prose quotes, for cross-checking
key = {
    "test_n": int(len(test_labels)), "test_tb": int(test_labels.sum()),
    "mis_any": int(len(mis)), "mis_all": int((mis["n_models_wrong"] == 4).sum()),
}
json.dump(key, open("report/key_numbers.json", "w"), indent=2)
print(key)
print("done")
