# Prompt: B.Tech. report (LaTeX, MNNIT format) + presentation deck

> Use this in Claude Code (or any assistant with file access) opened in the
> project folder, AFTER all experiments have finished, so every number is read
> from the final result files. Fill in the bracketed [STUDENT: ...] items
> yourself; they are deliberately not generated.

---

## Prompt (copy everything below this line)

You are helping me write the final B.Tech. project report and the presentation
deck for my project **"Hybrid Quantum–Classical Neural Networks for
Tuberculosis Detection from Chest X-rays"**. The repository in the current
folder contains all code (`src/`, `configs/`) and all experimental results
(`results/tb/`). Read the code and results before writing anything.

### Part A — Ground rules (apply to both deliverables)

1. **Every number must come from a file in `results/`.** Do not round
   differently in different places; use 2 decimal places for percentages and
   4 for F1/AUC unless the source table uses something else. If a number you
   want is not in a results file, say so in a `% TODO` comment instead of
   estimating it.
2. **Do not invent citations.** Only cite papers from the reference list in
   Part D, plus any you can verify exist (title, authors, venue, year). Never
   fabricate DOIs, page numbers or quotes. If unsure of a detail, omit it.
3. **Describe what the code actually does.** Architecture details,
   hyperparameters, splits and training settings must match `configs/config.py`
   and `src/*.py` exactly (e.g. 64×64 grayscale, 70/15/15 stratified split
   with `split_seed=42`, Adam, lr 1e-3, 15 epochs, batch 32, 4 qubits, depth 2,
   seeds 42/43/44). Do not describe features that do not exist.
4. **Report results honestly.** Where a difference is within seed-to-seed
   standard deviation or the McNemar p-value is ≥ 0.05, say it is not
   statistically significant. Do not call the hybrid "superior" unless the
   evidence in the files supports it; use "comparable", "consistently
   slightly higher", "more parameter-efficient", etc. as the data warrants.
   Include the limitations listed in Part C.
5. **Writing quality.** Write in clear, formal, specific technical English.
   Prefer concrete statements tied to this project ("the quanvolution layer
   evaluates 1,024 four-qubit circuits per 64×64 image") over generic ones
   ("quantum computing is a rapidly growing field"). No filler, no repeated
   summaries, no marketing language, no rhetorical questions, no bullet lists
   where a paragraph is expected. Define every symbol on first use. Use
   active voice where natural and first-person plural ("we") consistently.
6. **Leave clearly marked placeholders** for: name, registration number,
   supervisor, acknowledgements, and a short personal "contribution
   statement" — `[STUDENT: ...]`. Do not write these.

### Part B — Report (LaTeX, 35–45 pages compiled)

Use `thesis.tex` as the template **without changing its document class
(`mnnit`) or preamble commands** (`\title`, `\author`, `\supervisor`,
`\specialization`, `\beforepreface`, `\prefacesection`, `\afterpreface`,
`\chapter`, `\appendix`, `\bibliographystyle{acm}`, `\bibliography{references}`).
Keep the five chapter titles from the template. Split chapters into files
`chapters/ch1_introduction.tex` … `chapters/ch5_conclusion.tex` and
`\input` them. Put all BibTeX entries in `references.bib`. Put figures in
`figures/` (copy them from `results/tb/...`; do not regenerate or alter them).

Target page budget (11–12 pt, as the class sets it), total ≈ 40 pages:

**Front matter** — Title page, Certificate/Declaration (if the class
provides it), Preface (½ page, about this project, replacing template text),
Acknowledgements `[STUDENT]`, Abstract (≈250 words, with the key numbers),
Table of contents, List of figures, List of tables, List of abbreviations
(CNN, VQC, PQC, NISQ, QML, C→Q, Q→C, ROC-AUC, …).

**Ch. 1 Introduction (≈5 pp)**
- 1.1 Tuberculosis and chest radiography: burden, why automated screening
  matters (cite WHO Global TB Report).
- 1.2 Deep learning for chest X-ray screening (brief).
- 1.3 Quantum machine learning and the NISQ setting (Preskill).
- 1.4 Problem statement: can quantum circuits add value to a CNN classifier,
  and in which arrangement (classical→quantum vs quantum→classical)?
- 1.5 Objectives (numbered, measurable).
- 1.6 Contributions of this work (numbered list of what was actually built
  and measured).
- 1.7 Report organisation.

**Ch. 2 Related Work (≈7 pp)**
- 2.1 CNNs and transfer learning for TB/CXR (Rahman et al. 2020, ResNet).
- 2.2 Quantum computing background needed later: qubit, Bloch sphere,
  rotation gates RX/RY/RZ, CNOT, measurement and expectation values ⟨Z⟩,
  statevector vs density-matrix simulation. Include the key equations.
- 2.3 Variational / parameterised quantum circuits and circuit-centric
  classifiers (Schuld et al.), parameter-shift rule (Mitarai; Schuld 2019).
- 2.4 Data encoding: angle vs amplitude encoding, with equations and
  qubit/feature counts.
- 2.5 Hybrid architectures: classical→quantum transfer learning (Mari et
  al.), quanvolutional networks (Henderson et al.).
- 2.6 Trainability and circuit quality: barren plateaus (McClean; Cerezo),
  expressibility and entangling capability (Sim et al.), Meyer–Wallach.
- 2.7 Research gap this project addresses (1 paragraph).

**Ch. 3 Proposed Work (≈9 pp)**
- 3.1 System overview figure (draw with TikZ: data → preprocessing →
  {CNN | ResNet18 | quanvolution} → {classical | quantum} head → output).
- 3.2 Dataset and preprocessing: Kaggle TB Chest X-ray Database counts
  (3,500 normal / 700 TB), grayscale, 64×64 resize, normalisation, stratified
  split sizes (2,940 / 630 / 630), caching.
- 3.3 Classical baseline CNN: layer table with shapes and parameter counts.
- 3.4 C→Q hybrid: CNN → Linear(32→4) → tanh·π → AngleEmbedding(RY) →
  StronglyEntanglingLayers (depth L) → ⟨Z_i⟩ → Linear(4→1) → sigmoid.
  Give the circuit diagram (from `results/tb/quantum_analysis/circuits.txt`,
  redraw with `quantikz` or include as verbatim), the state after encoding,
  the weight tensor shape (L, n, 3), and why tanh·π.
- 3.5 Q→C hybrid: quanvolution — 2×2 patches, stride 2, RY(πp) encoding,
  frozen RandomLayers, 4 output channels, 32×32 feature map, precomputation;
  then the CNN.
- 3.6 Fair classical controls and why each exists: matched classical head
  (157 vs 161 head parameters), random patch map, average pooling, learned
  2×2 convolution.
- 3.7 Transfer-learning variant: ResNet18 backbone, grayscale→3 channels,
  reduced learning rate (×0.1) for pretrained layers.
- 3.8 Circuit variants studied: angle/amplitude encoding; strong/basic/no
  entanglement; depth; qubits.
- 3.9 Training procedure and gradients: BCE loss, Adam, backprop through the
  simulator vs parameter-shift on hardware; best-validation-loss checkpoint
  in the low-data study.
- 3.10 Evaluation protocol: metrics (accuracy, precision, recall, F1,
  ROC-AUC — with formulas), seeds, mean ± std, McNemar's exact test, model
  selection on validation only.

**Ch. 4 Experimental Setup and Results Analysis (≈13 pp)** — one subsection
per experiment; each with: setup (1 paragraph), table, figure, and an
analysis paragraph that interprets the numbers (not just repeats them).
- 4.1 Environment: Python, PyTorch, PennyLane versions (read them with
  `pip show`), CPU-only hardware `[STUDENT: laptop model/CPU/RAM]`,
  training times.
- 4.2 Baseline vs C→Q hybrid — `results/tb/baseline.json`, `hybrid.json`,
  confusion matrices.
- 4.3 Circuit ablation — `results/tb/ablation_summary.md` and the four
  `ablation_*.png` plots (note the zoomed y-axis: the whole qubit plot spans
  only ≈10 test images).
- 4.4 Backbone × head 4-way comparison with 3 seeds —
  `results/tb/comparison/summary.md`, `roc_curves.png`, confusion matrices.
- 4.5 Main hybrid study and low-data regime —
  `results/tb/hybrid_study/summary.md`, `f1_vs_data.png`,
  `auc_vs_data.png`, `accuracy_vs_data.png`, McNemar table. Discuss: C→Q vs
  matched head; Q→C vs each of its three classical controls; parameter
  efficiency (≈34k vs ≈132k parameters); instability of bottleneck heads at
  small data sizes.
- 4.6 Quantum circuit analysis — `results/tb/quantum_analysis/summary.md`:
  expressibility and entangling capability plots, barren plateau plot
  (gradient variance vs qubits, and what it implies for scaling), noise and
  shot robustness plots, gradient-check result.
- 4.7 Error analysis — `results/tb/comparison/errors_*.png`,
  `misclassified.csv`: how many errors are shared by all models, FN vs FP,
  clinical meaning of missed TB.
- 4.8 Comparison with published work (Rahman et al. 2020) — state clearly
  that resolution, split and preprocessing differ, so the comparison is
  indicative only.
- 4.9 Discussion: what the evidence does and does not show.

**Ch. 5 Conclusion and Future Work (≈3 pp)** — numbered findings with their
numbers; limitations (Part C); future work: trainable quanvolution, data
re-uploading circuits, quantum kernels (Havlíček et al.), real-hardware runs
with error mitigation, higher resolution, external validation dataset,
Grad-CAM explainability.

**Appendix (≈3–5 pp)** — A: full hyperparameter table from
`configs/config.py`; B: per-seed tables from the `runs.csv` files; C: all
circuit diagrams; D: how to reproduce (commands from the README).

**LaTeX requirements**
- Every figure/table has a caption, a `\label`, and is referenced in the text
  with `\ref` before it appears. Use `booktabs` tables and `siunitx` if the
  class allows extra packages (load them in `thesis.tex` only if they compile
  with the `mnnit` class).
- Use `\cite` for every factual claim from the literature.
- Must compile with `pdflatex → bibtex → pdflatex → pdflatex` with **zero
  errors, zero undefined references/citations, and no overfull hboxes wider
  than 5 pt**. Actually run the build, read the `.log`, fix every warning
  that matters, and report the final page count.

### Part C — Limitations that must appear (Ch. 4.9 and Ch. 5)
- Quantum circuits are classically simulated (4–8 qubits); no real hardware.
- Images downscaled to 64×64; single public dataset; no external validation.
- Possible dataset-source bias: normal and TB images come partly from
  different sources, so models may exploit acquisition differences.
- Many accuracy differences are within seed variance; per-seed McNemar
  p-values are mostly ≥ 0.05 for the C→Q comparisons.
- Training-time measurements were taken on a shared laptop CPU and vary
  between runs.
- The quanvolution filter is random and frozen, not trained.

### Part D — Reference list (verify each before use; add more only if verified)
- Rahman, T. et al. "Reliable Tuberculosis Detection Using Chest X-Ray With
  Deep Learning, Segmentation and Visualization." IEEE Access 8 (2020).
- World Health Organization. Global Tuberculosis Report (use the latest
  edition you can verify).
- He, K., Zhang, X., Ren, S., Sun, J. "Deep Residual Learning for Image
  Recognition." CVPR 2016.
- Preskill, J. "Quantum Computing in the NISQ era and beyond." Quantum 2, 79
  (2018).
- Schuld, M., Bocharov, A., Svore, K., Wiebe, N. "Circuit-centric quantum
  classifiers." Physical Review A 101, 032308 (2020).
- Mitarai, K., Negoro, M., Kitagawa, M., Fujii, K. "Quantum circuit
  learning." Physical Review A 98, 032309 (2018).
- Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., Killoran, N.
  "Evaluating analytic gradients on quantum hardware." Physical Review A 99,
  032331 (2019).
- Mari, A., Bromley, T., Izaac, J., Schuld, M., Killoran, N. "Transfer
  learning in hybrid classical-quantum neural networks." Quantum 4, 340
  (2020).
- Henderson, M., Shakya, S., Pradhan, S., Cook, T. "Quanvolutional neural
  networks: powering image recognition with quantum circuits." Quantum
  Machine Intelligence 2 (2020).
- McClean, J. et al. "Barren plateaus in quantum neural network training
  landscapes." Nature Communications 9, 4812 (2018).
- Cerezo, M. et al. "Cost function dependent barren plateaus in shallow
  parametrized quantum circuits." Nature Communications 12, 1791 (2021).
- Sim, S., Johnson, P., Aspuru-Guzik, A. "Expressibility and entangling
  capability of parameterized quantum circuits for hybrid quantum-classical
  algorithms." Advanced Quantum Technologies 2, 1900070 (2019).
- Meyer, D., Wallach, N. "Global entanglement in multiparticle systems."
  Journal of Mathematical Physics 43, 4273 (2002).
- Bergholm, V. et al. "PennyLane: Automatic differentiation of hybrid
  quantum-classical computations." arXiv:1811.04968.
- Paszke, A. et al. "PyTorch: An Imperative Style, High-Performance Deep
  Learning Library." NeurIPS 2019.
- Kingma, D., Ba, J. "Adam: A Method for Stochastic Optimization." ICLR 2015.
- McNemar, Q. "Note on the sampling error of the difference between
  correlated proportions or percentages." Psychometrika 12 (1947).
- Havlíček, V. et al. "Supervised learning with quantum-enhanced feature
  spaces." Nature 567 (2019). (future work)
- Pérez-Salinas, A. et al. "Data re-uploading for a universal quantum
  classifier." Quantum 4, 226 (2020). (future work)

### Part E — Presentation deck (22–28 slides, .pptx)

Build it with a Python script `docs/make_slides.py` using `python-pptx`
(16:9, one consistent clean template, ≥ 20 pt body text, max ~6 lines per
slide, real figures from `results/tb/` embedded at full width, slide numbers,
speaker notes on every slide with what to say in ~45–60 s). Run the script
and confirm the file opens and every image is present.

Slide plan:
1. Title (project, `[STUDENT]` names/reg. nos., guide, institute, date)
2. Outline
3. Motivation: TB burden and screening
4. Problem statement and objectives
5. Quantum computing in 1 slide: qubit, gates, measurement
6. Variational quantum circuits and the parameter-shift rule
7. Angle vs amplitude encoding
8. Related work (table: work — method — result)
9. Dataset and preprocessing (counts, split, sample images)
10. System overview diagram
11. Classical baseline CNN
12. C→Q hybrid architecture + circuit diagram
13. Q→C hybrid: quanvolution
14. Fair controls: why they matter
15. Experimental setup and evaluation protocol
16. Result: baseline vs hybrid (table + confusion matrices)
17. Circuit ablation (depth / qubits / entanglement / encoding)
18. 4-way comparison incl. ResNet18 (table + ROC)
19. Low-data study (F1 vs data plot)
20. Statistical significance (McNemar table, plain-language reading)
21. Expressibility & entangling capability
22. Barren plateaus
23. Noise & shot robustness (NISQ readiness)
24. Error analysis (misclassified images)
25. Key findings (numbered, with numbers)
26. Limitations
27. Future work
28. Thank you / Questions

### Part F — Final checks (do all, then report the results to me)
- Cross-check every number in the report and slides against the source file
  (list any mismatch you fixed).
- Confirm architecture/hyperparameter statements against the code.
- Confirm every `\cite` key exists in `references.bib` and every entry is
  real.
- Report: compiled page count, number of figures/tables, slide count, and
  the list of `[STUDENT: ...]` placeholders I still need to fill.
