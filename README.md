# Hybrid Quantum–Classical Neural Networks for Tuberculosis Detection

Binary classification of chest X-rays (Normal vs. Tuberculosis) comparing a
classical CNN with two hybrid quantum–classical architectures, each checked
against fair classical controls, plus an analysis of the quantum circuits
themselves. Quantum circuits are simulated with PennyLane; models are built
in PyTorch.

| Architecture | Idea |
|---|---|
| **Classical CNN** (baseline) | 2× (Conv–ReLU–MaxPool) → Dense(32) → Linear → sigmoid |
| **C→Q hybrid** (classical → quantum) | CNN features → Linear(32→4) → tanh·π → 4-qubit variational circuit (angle encoding + StronglyEntanglingLayers) → ⟨Z⟩ ×4 → Linear → sigmoid |
| **Q→C hybrid** (quantum → classical) | Quanvolution: each 2×2 pixel patch → 4-qubit circuit (RY encoding + frozen RandomLayers) → 4-channel 32×32 quantum feature map → CNN |
| ResNet18 variants | ImageNet-pretrained ResNet18 backbone with classical or quantum head |

Fair controls: a **matched classical head** (same bottleneck, quantum layer
replaced by Linear(4,4)+tanh; 157 vs 161 head parameters) for C→Q, and three
patch-level controls for Q→C (**random patch map**, **2×2 average pooling**,
**learned 2×2 convolution**), all ≈34k parameters.

## Dataset

[Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
— 3,500 normal, 700 TB images. Converted to grayscale, resized to 64×64,
stratified 70/15/15 split (2,940 / 630 / 630) with a fixed `split_seed`.

Download and extract so that `data/tb_raw/TB_Chest_Radiography_Database/{Normal,Tuberculosis}/`
exists (or set `tb_root` in `configs/config.py`). Public datasets can be
fetched without a Kaggle token:

```
curl -L -o data/tb_raw/tb.zip https://www.kaggle.com/api/v1/datasets/download/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
```

PneumoniaMNIST (the project's original dataset) is still supported with
`--dataset pneumoniamnist`.

## Setup

```
pip install -r requirements.txt
```

CPU is enough. All runs are seeded and reproducible.

## Running

| Command | What it does | Time (laptop CPU) |
|---|---|---|
| `python -m src.train --model baseline` / `--model hybrid` | Single baseline / C→Q hybrid run | ~0.5 / ~1.5 min |
| `python -m src.ablation` | Circuit ablation: depth, qubits × encoding, entanglement | ~15 min |
| `python -m src.compare` | {CNN, ResNet18} × {classical, quantum}, 3 seeds, ROC + error analysis | ~1 h |
| `python -m src.hybrid_study` | 7 models × 5 training-set sizes × 3 seeds, McNemar tests | ~30 min |
| `python -m src.quantum_analysis` | Expressibility, entangling capability, barren plateaus, noise/shots, gradient check | ~10 min |

Every script accepts `--subset` for a fast smoke test.

## Results (test set, 630 images; mean ± std over seeds 42/43/44)

**Full training data**

| Model | Params | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Classical CNN | 132,385 | 97.67 ± 0.33% | 0.924 | 0.930 ± 0.009 | 0.9966 |
| CNN + matched classical head | 132,509 | 97.72 ± 0.40% | 0.908 | 0.930 ± 0.012 | 0.9959 |
| **C→Q hybrid** | 132,513 | **98.41 ± 0.32%** | 0.943 | **0.952 ± 0.010** | 0.9980 |
| Q→C hybrid (quanvolution) | 34,297 | 97.88 ± 0.33% | 0.927 | 0.936 ± 0.010 | 0.9954 |
| Random patch + CNN | 34,297 | 90.53 ± 0.87% | 0.521 | 0.645 ± 0.054 | 0.9336 |
| Avg-pool + CNN | 34,081 | 98.15 ± 0.09% | 0.956 | 0.945 ± 0.002 | 0.9956 |
| Learned 2×2 conv + CNN | 34,317 | 98.31 ± 0.72% | 0.952 | 0.950 ± 0.020 | 0.9956 |
| ResNet18 + classical | 11.19 M | 99.21 ± 0.57% | 0.965 | 0.976 ± 0.018 | 0.9991 |
| **ResNet18 + quantum** | 11.19 M | **99.47 ± 0.18%** | 0.975 | **0.984 ± 0.006** | 0.9961 |

**Findings**

1. The C→Q hybrid has the highest mean accuracy/F1 among the small-CNN
   models and the quantum head also helps on ResNet18, with lower
   seed-to-seed variance. The gain (≈0.7 pp over the matched classical head)
   is consistent across seeds but **not statistically significant** per seed
   (McNemar p = 0.09, 0.15, 1.00).
2. The Q→C hybrid clearly beats a random classical patch map (p < 0.001),
   but **matches, not beats**, simple average pooling and a learned 2×2
   convolution (p ≥ 0.06). Its parameter efficiency (≈4× fewer parameters
   than the full CNN at equal accuracy) comes from the 2×2 downsampling, not
   specifically from the quantum filter.
3. Low data (2% = 58 images): Q→C and avg-pool are the best (F1 ≈ 0.67);
   Q→C is the most stable (std 0.020 vs 0.138). Bottleneck heads — both the
   quantum head and its matched classical control — are unstable with very
   little data.
4. Circuit ablation: depth (1–4), qubit count (2–8) and entanglement pattern
   change accuracy by ≤ 1–2 test images; angle encoding is consistently
   better than amplitude encoding.
5. Circuit analysis: entangling layers reach near-Haar expressibility by
   depth 2–3 (KL ≈ 0.004–0.008) with Meyer–Wallach entanglement ≈ 0.82–0.87;
   the no-entanglement control stays at KL ≈ 0.2 and zero entanglement.
   Gradient variance falls ≈ 250× from 2 to 10 qubits (barren-plateau
   scaling), supporting small circuits.
6. NISQ realism: the trained C→Q model keeps 98.2–98.6% accuracy with only
   10 measurement shots and under depolarizing noise up to p = 0.2 per qubit
   per layer; parameter-shift (hardware) gradients match backprop to 1e-15.

**Limitations.** Simulated qubits only; 64×64 images; one public dataset
whose normal and TB images partly come from different sources (possible
shortcut learning); many differences are within seed variance; training
times measured on a shared laptop CPU; quanvolution filter is random and
frozen.

Detailed tables and plots: `results/tb/{ablation_summary.md, comparison/,
hybrid_study/, quantum_analysis/}`.

## Repository layout

```
configs/config.py        every hyperparameter (single source of truth)
src/data.py              TB / PneumoniaMNIST loading, stratified split, low-data subsets, feature caching
src/models.py            CNN & ResNet18 backbones; classical, matched-classical and quantum heads
src/quanvolution.py      quantum convolution filter + classical patch controls
src/train.py             training loop (single runs), circuit drawing
src/evaluate.py          metrics, confusion matrices, ROC curves
src/ablation.py          circuit ablation (depth, qubits, encoding, entanglement)
src/compare.py           4-way backbone × head comparison, error analysis
src/hybrid_study.py      main classical vs C→Q vs Q→C study with controls + low-data sweep
src/quantum_analysis.py  expressibility, entangling capability, barren plateaus, noise, shots
results/tb/              all TB results;  results/pneumoniamnist/  original results
docs/                    report/slides generation prompt
```
