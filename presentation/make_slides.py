"""
Build the project presentation (16:9 .pptx) from the result files.

    python presentation/make_slide_figures.py   # charts used on the slides
    python presentation/make_slides.py

20 slides: title, 17 content slides, references, thank you.
Numbers come from results/ (or match the report). Plain language, large text,
and the team and supervisor names on every slide.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

FIG = "report/figures"
SFIG = "presentation/figures"
RES = "results/tb"
OUT = os.environ.get("DECK_OUT", "presentation/TB_Hybrid_Quantum_Presentation.pptx")
REPO = "github.com/realsaksham/Mini-project-7th-sem"

STUDENTS = [("Saksham Tiwari", "20233243"), ("Prajjwal Singh", "20233213"),
            ("Madipelli Rithvik", "20233176"), ("Sanchari Ray", "20233250")]
GUIDE = "Dr. Rajitha B."
FOOTER = "  |  ".join(n for n, _ in STUDENTS) + f"   ·   Guide: {GUIDE}"

INK = RGBColor(0x1B, 0x1B, 0x1F)
MUTED = RGBColor(0x55, 0x55, 0x5E)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)     # classical blue
QUANTUM = RGBColor(0xEB, 0x68, 0x34)    # quantum orange
LIGHT = RGBColor(0xF3, 0xF5, 0xF9)
PALE_ORANGE = RGBColor(0xFD, 0xEB, 0xE3)
PALE_BLUE = RGBColor(0xE6, 0xEF, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
NAVY = RGBColor(0x14, 0x2B, 0x4D)
SOFT = RGBColor(0xC9, 0xD6, 0xEA)
FONT = "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, H = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]

L, T = Inches(0.55), Inches(1.55)       # content origin
CW, CH = Inches(12.25), Inches(5.2)     # content box (ends at 6.75 in)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def text(slide, x, y, w, h, s, size=22, bold=False, color=INK, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    for k, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, FONT
    return box


def bullets(slide, x, y, w, h, items, size=24, gap=14):
    """Items starting with '- ' are sub-points (smaller, grey)."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for k, item in enumerate(items):
        sub = item.startswith("- ")
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.space_after = Pt(gap - 6 if sub else gap)
        r = p.add_run()
        r.text = ("      " + item[2:]) if sub else ("•  " + item)
        r.font.name = FONT
        r.font.size = Pt(size - 3 if sub else size)
        r.font.color.rgb = MUTED if sub else INK
    return box


def card(slide, x, y, w, h, title, lines, color=ACCENT, size=22):
    bg = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    bg.adjustments[0] = 0.05
    bg.fill.solid(); bg.fill.fore_color.rgb = LIGHT; bg.line.fill.background()
    hdr = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, Inches(0.65))
    hdr.fill.solid(); hdr.fill.fore_color.rgb = color; hdr.line.fill.background()
    text(slide, x + Inches(0.15), y + Inches(0.07), w - Inches(0.3), Inches(0.5), title,
         size=23, bold=True, color=WHITE)
    bullets(slide, x + Inches(0.15), y + Inches(0.85), w - Inches(0.3), h - Inches(0.95), lines,
            size=size, gap=12)


def note_box(slide, x, y, w, h, s, size=22, color=QUANTUM, fill=PALE_ORANGE):
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    box.fill.solid(); box.fill.fore_color.rgb = fill; box.line.fill.background()
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(0.1), h)
    bar.fill.solid(); bar.fill.fore_color.rgb = color; bar.line.fill.background()
    text(slide, x + Inches(0.3), y, w - Inches(0.45), h, s, size=size, bold=True,
         anchor=MSO_ANCHOR.MIDDLE)


def picture(slide, path, x, y, max_w, max_h):
    """Insert an image scaled to fit (max_w, max_h), centred in that box."""
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(path, x + (max_w - w) // 2, y + (max_h - h) // 2, w, h)


def new_slide(title, notes):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), H)
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
    text(s, Inches(0.55), Inches(0.38), Inches(12.3), Inches(0.8), title, size=32, bold=True)
    line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.2), Inches(1.2), Inches(0.05))
    line.fill.solid(); line.fill.fore_color.rgb = QUANTUM; line.line.fill.background()
    text(s, Inches(0.55), H - Inches(0.5), Inches(11), Inches(0.35), FOOTER, size=13, color=MUTED)
    # Empty placeholder, filled with the slide number at the end.
    text(s, W - Inches(1.4), H - Inches(0.5), Inches(1.0), Inches(0.35), "", size=13,
         color=MUTED, align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def table(slide, x, y, w, header, rows, col_w=None, size=20, highlight_rows=(), row_h=0.6):
    nr, nc = len(rows) + 1, len(header)
    tbl = slide.shapes.add_table(nr, nc, x, y, w, Inches(row_h) * nr).table
    if col_w:
        for i, cw in enumerate(col_w):
            tbl.columns[i].width = Inches(cw)
    for r in range(nr):
        tbl.rows[r].height = Inches(row_h)
        for c in range(nc):
            cell = tbl.cell(r, c)
            cell.text = str(header[c] if r == 0 else rows[r - 1][c])
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            para = cell.text_frame.paragraphs[0]
            para.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            for run in para.runs:
                run.font.size, run.font.name = Pt(size), FONT
                run.font.bold = r == 0 or (r - 1) in highlight_rows
                run.font.color.rgb = WHITE if r == 0 else INK
            cell.fill.solid()
            cell.fill.fore_color.rgb = (ACCENT if r == 0 else
                                        PALE_ORANGE if (r - 1) in highlight_rows else
                                        LIGHT if r % 2 == 0 else WHITE)
    return tbl


def team_block(slide, x, y, size=22):
    lines = "\n".join(f"{n}  ({reg})" for n, reg in STUDENTS)
    text(slide, x, y, Inches(6.0), Inches(2.0), lines, size=size, color=WHITE)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
comp = pd.read_csv(f"{RES}/comparison/summary.csv", index_col=0)
study = pd.read_csv(f"{RES}/hybrid_study/summary.csv")
full = study[study.train_fraction == 1.0].set_index("model")
hw = pd.read_csv(f"{RES}/quantum_analysis/hardware_realism.csv")

CQ = "C->Q hybrid (CNN + quantum head)"
QC = "Q->C hybrid (quanvolution + CNN)"
MATCHED = "CNN + matched classical head"
acc = lambda m: 100 * full.loc[m].accuracy_mean
cnn_c, cnn_q = comp.loc["CNN + classical"], comp.loc["CNN + quantum"]
rn_c, rn_q = comp.loc["ResNet18 + classical"], comp.loc["ResNet18 + quantum"]
exact = 100 * hw[hw.setting == "exact"].accuracy.iloc[0]

# ===========================================================================
# 1. Title
# ===========================================================================
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
text(s, Inches(0.8), Inches(0.8), Inches(11.7), Inches(2.0),
     "Hybrid Quantum–Classical Neural Networks for Tuberculosis Detection from Chest X-rays",
     size=40, bold=True, color=WHITE)
stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(2.95), Inches(1.6), Inches(0.07))
stripe.fill.solid(); stripe.fill.fore_color.rgb = QUANTUM; stripe.line.fill.background()
text(s, Inches(0.8), Inches(3.2), Inches(11.5), Inches(0.5),
     "B.Tech. Mini Project, Computer Science & Engineering", size=22, color=SOFT)
text(s, Inches(0.8), Inches(3.95), Inches(6), Inches(0.45), "Presented by", size=18, bold=True, color=SOFT)
team_block(s, Inches(0.8), Inches(4.4))
text(s, Inches(7.3), Inches(3.95), Inches(5.5), Inches(0.45), "Under the guidance of", size=18,
     bold=True, color=SOFT)
text(s, Inches(7.3), Inches(4.4), Inches(5.5), Inches(2.2),
     f"{GUIDE}\nDepartment of CSE\nMNNIT Allahabad, Prayagraj", size=22, color=WHITE)
s.notes_slide.notes_text_frame.text = (
    "Good morning. We are Saksham, Prajjwal, Rithvik and Sanchari, and our project was guided by "
    "Dr. Rajitha B. We tried to find out whether adding a small quantum circuit to a normal "
    "deep-learning model helps it detect tuberculosis in chest X-rays.")

# ===========================================================================
# 2. Why TB
# ===========================================================================
s = new_slide("Why tuberculosis screening matters", notes=(
    "TB is still the biggest killer among single infectious diseases. India has the most cases. "
    "A chest X-ray is a cheap way to screen people, but someone trained has to read every image, and "
    "there are not enough radiologists. Software that flags likely TB cases can help, and the WHO "
    "already recommends it. On the right you can see how similar normal and TB X-rays can look."))
bullets(s, L, T + Inches(0.15), Inches(6.2), CH, [
    "TB kills about 1.25 million people a year",
    "India has the largest number of cases",
    "Chest X-rays are cheap and quick",
    "But every X-ray needs an expert to read it",
    "Software can sort out the likely TB cases first"], size=26, gap=22)
picture(s, f"{SFIG}/samples_labeled.png", Inches(6.9), T, Inches(6.0), CH)

# ===========================================================================
# 3. Question and answer
# ===========================================================================
s = new_slide("What we wanted to find out", notes=(
    "Many papers say a quantum model did better, but they compare it with a classical model that is "
    "different in size, and they train it only once. So we asked a fair question: if we add a small "
    "quantum circuit to a CNN, does it really help? And does it matter where we put it? Here is our "
    "answer in short. The C to Q model, with the quantum part after the CNN, helped a little, but the difference is "
    "too small to be sure. The Q to C model, with the quantum part before the CNN, saved parameters, but a simple classical "
    "trick did the same."))
note_box(s, L, T, CW, Inches(1.25),
         "Does adding a small quantum circuit to a CNN help it detect TB?\n"
         "Does it matter if the circuit goes after the CNN (C→Q) or before it (Q→C)?",
         size=24, color=ACCENT, fill=PALE_BLUE)
text(s, L, T + Inches(1.55), CW, Inches(0.5), "Our short answer", size=24, bold=True, color=QUANTUM)
bullets(s, L, T + Inches(2.15), CW, Inches(3.0), [
    "C→Q (quantum circuit after the CNN): a small gain (about 0.7%), too small to be sure it is real",
    "Q→C (quantum circuit before the CNN): 4 times fewer parameters, but simple image shrinking does the same",
    "The quantum circuits still work with noise, as on real quantum computers"],
    size=25, gap=14)

# ===========================================================================
# 4. Quantum basics
# ===========================================================================
s = new_slide("Quantum computing basics", notes=(
    "A normal bit is 0 or 1. A qubit can be a mix of both. We change qubits with rotation gates and "
    "link them together with a CNOT gate; this linking is called entanglement. When we measure, we "
    "get a 0 or a 1, so we repeat the circuit many times and take the average. In a quantum neural "
    "network the rotation angles are the weights, and we train them just like normal weights. "
    "Today's quantum computers are small and noisy, so only small circuits are practical."))
card(s, L, T, Inches(6.0), CH, "The building blocks", [
    "Qubit: can be 0, 1, or a mix of both",
    "Rotation gate: turns a qubit by an angle",
    "CNOT gate: links two qubits (entanglement)",
    "Measuring gives 0 or 1, so we repeat and average",
    "n qubits hold 2ⁿ numbers"], size=24)
card(s, Inches(6.8), T, Inches(6.0), CH, "How a quantum circuit learns", [
    "Put the input numbers in as angles",
    "Apply rotations with trainable angles",
    "Measure and use the result",
    "Train the angles like normal weights",
    "Real quantum computers are small and noisy"], color=QUANTUM, size=24)

# ===========================================================================
# 5. Dataset
# ===========================================================================
s = new_slide("Our dataset", notes=(
    "We used a public Kaggle dataset with 4,200 chest X-rays: 3,500 normal and 700 with TB. Because "
    "there are five normal images for every TB image, a model that always says normal would already "
    "be 83 percent accurate, so we also look at F1 score and recall. We made the images black and "
    "white and shrank them to 64 by 64 pixels so that the quantum simulation runs on a laptop. Every "
    "model is tested on the same 630 images."))
bullets(s, L, T + Inches(0.15), Inches(6.3), CH, [
    "TB Chest X-ray Database (Kaggle)",
    "3,500 normal and 700 TB images",
    "Always saying 'normal' already gives 83%",
    "- so we also use F1 score and recall",
    "Images shrunk to 64 × 64 pixels, grayscale",
    "Same 630 test images for every model"], size=25, gap=16)
table(s, Inches(7.3), T + Inches(0.4), Inches(5.5), ["Part", "Normal", "TB", "Total"],
      col_w=[1.9, 1.3, 1.0, 1.3], rows=[
          ["Training", "2,450", "490", "2,940"], ["Validation", "525", "105", "630"],
          ["Testing", "525", "105", "630"], ["Total", "3,500", "700", "4,200"]],
      size=22, highlight_rows=(3,), row_h=0.8)

# ===========================================================================
# 6. Three models
# ===========================================================================
s = new_slide("The three models we compared", notes=(
    "We compared three models. The first is a small normal CNN. The second, C to Q, keeps the same CNN but "
    "replaces its last layer with a 4-qubit quantum circuit. The third, Q to C, puts a quantum filter at the "
    "start, which looks at small 2 by 2 pieces of the image, and then uses a CNN. Blue parts are "
    "classical and orange parts are quantum."))
picture(s, f"{FIG}/architectures.png", L, T, CW, Inches(4.5))
text(s, L, Inches(6.15), CW, Inches(0.55),
     "(a) Classical CNN    (b) C→Q: classical → quantum    (c) Q→C: quantum → classical", size=22, color=MUTED,
     align=PP_ALIGN.CENTER)

# ===========================================================================
# 7. CNN + quantum head
# ===========================================================================
s = new_slide("C→Q hybrid: CNN + quantum classifier", notes=(
    "Here the CNN first turns the X-ray into 32 numbers. We shrink these to 4 numbers and put each "
    "one into a qubit as a rotation angle. Then two layers of trainable rotations and CNOT gates mix "
    "the qubits. We measure each qubit and a last small layer gives the probability of TB. The "
    "quantum part has only 24 trainable angles, and the whole model is almost the same size as the "
    "normal CNN."))
picture(s, f"{FIG}/circuit_cq_strong.png", L, T + Inches(0.2), Inches(7.0), Inches(4.8))
bullets(s, Inches(7.9), T + Inches(0.15), Inches(5.0), CH, [
    "CNN turns the X-ray into 32 numbers",
    "Shrunk to 4 numbers, one per qubit",
    "2 layers of trainable rotations",
    "- only 24 trainable angles",
    "Measure, then output P(TB)",
    "Same size as the normal CNN"], size=24, gap=15)

# ===========================================================================
# 8. Quantum filter + CNN
# ===========================================================================
s = new_slide("Q→C hybrid: quantum filter + CNN", notes=(
    "In this model the quantum circuit comes first. It looks at every 2 by 2 block of pixels, puts "
    "the 4 pixels into 4 qubits, runs a fixed random circuit and measures, giving 4 new images. That "
    "is 1,024 circuits per X-ray, so we computed them once and saved them. The new images are half "
    "the size, so the CNN after it needs about 4 times fewer parameters. You can see the 4 quantum "
    "output images here; the white box at the bottom is a marker on the original X-ray."))
picture(s, f"{FIG}/quanv_feature_maps.png", L, T, CW, Inches(2.6))
bullets(s, L, T + Inches(2.9), Inches(7.6), Inches(2.4), [
    "Looks at every 2 × 2 block of pixels",
    "4 pixels → 4 qubits → random circuit → 4 outputs",
    "1,024 circuits per X-ray, computed once",
    "Half-size images → 4× fewer parameters"], size=23, gap=12)
picture(s, f"{FIG}/circuit_quanv.png", Inches(8.4), T + Inches(2.8), Inches(4.4), Inches(2.4))

# ===========================================================================
# 9. Fair test + training
# ===========================================================================
s = new_slide("How we made the comparison fair", notes=(
    "To know if the quantum part really helps, each quantum model gets a classical twin that is the "
    "same in every way except the quantum circuit. For the quantum classifier, the twin uses a small "
    "classical layer of the same size. For the quantum filter we tried three classical filters: a "
    "random one, simple average pooling, and a learned one. All models were trained the same way, "
    "three times each with different random seeds, and we used a statistical test to check if "
    "differences are real."))
card(s, L, T, Inches(6.6), CH, "Classical twins", [
    "C→Q vs a classical layer of the same size",
    "Q→C vs three classical filters:",
    "- random, average pooling, learned",
    "Same data, same settings for all",
    "No extra tuning for any model"], size=24)
card(s, Inches(7.35), T, Inches(5.45), CH, "Training and testing", [
    "Adam optimiser, 15 epochs",
    "Each model trained 3 times",
    "Laptop CPU, no GPU",
    "McNemar test: is a difference real?",
    "Over 150 training runs in total"], color=QUANTUM, size=24)

# ===========================================================================
# 10. Circuit settings
# ===========================================================================
s = new_slide("Result 1: circuit settings hardly matter", notes=(
    "First we changed one thing in the quantum circuit at a time: the number of layers, the number of "
    "qubits, the type of linking between qubits, and the way the input is loaded. All bars use the "
    "same scale. Most settings land between 98.3 and 98.6 percent, which is a difference of just "
    "two test images. The only clear result is that loading the input as angles works better than "
    "the other method, called amplitude encoding."))
picture(s, f"{SFIG}/ablation_bars.png", L, T, CW, Inches(4.45))
note_box(s, L, Inches(6.05), CW, Inches(0.7),
         "Most settings differ by only 1 or 2 test images.  Angle input works best.",
         size=22, color=ACCENT, fill=PALE_BLUE)

# ===========================================================================
# 11. Backbone vs head
# ===========================================================================
s = new_slide("Result 2: a better CNN matters more than the quantum part", notes=(
    "Next we tried the quantum classifier on two CNNs: our small CNN and a big pretrained one called "
    "ResNet18. With both, the quantum head was a little better on average and more stable. But "
    "switching to the bigger CNN helped about three times more than switching to the quantum head."))
picture(s, f"{SFIG}/backbone_head.png", L, T, Inches(7.4), CH)
bullets(s, Inches(8.2), T + Inches(0.3), Inches(4.7), CH, [
    f"C→Q head: +{100 * (cnn_q.accuracy_mean - cnn_c.accuracy_mean):.2f}% on the small CNN",
    f"C→Q head: +{100 * (rn_q.accuracy_mean - rn_c.accuracy_mean):.2f}% on ResNet18",
    f"Bigger CNN: +{100 * (rn_c.accuracy_mean - cnn_c.accuracy_mean):.2f}%",
    f"Best model: ResNet18 + quantum head, {100 * rn_q.accuracy_mean:.2f}%"], size=24, gap=20)

# ===========================================================================
# 12. Main comparison
# ===========================================================================
s = new_slide("Result 3: C→Q and Q→C against their classical twins", notes=(
    f"This is our main result. On the left, the CNN with a quantum head got {acc(CQ):.2f} percent, "
    f"about 0.7 points better than its classical twin, and it was ahead in all three runs. On the "
    f"right, the quantum filter model matched the normal CNN with 4 times fewer parameters. But "
    f"simple average pooling and a learned classical filter did just as well. The random filter "
    f"did badly because it destroys image detail."))
picture(s, f"{SFIG}/main_study_bars.png", L, T, CW, Inches(4.45))
note_box(s, L, Inches(6.05), CW, Inches(0.7),
         "C→Q: +0.7% over its twin.   Q→C: same as average pooling.", size=22)

# ===========================================================================
# 12b. All model metrics in one table
# ===========================================================================
s = new_slide("All models at a glance", notes=(
    "This table puts every model's numbers in one place. Each value is the average of three training "
    "runs on the same 630 test images, using all the training data. The top seven rows are the main "
    "study: the normal CNN, C to Q and its classical twin, and Q to C with its three classical "
    "filters. The last two rows use the big pretrained ResNet18. Highlighted rows are our quantum "
    "models. Recall is the share of TB cases the model catches, which matters most for screening."))
METRIC_ROWS = [
    ("Classical CNN", "Classical CNN", full),
    ("CNN + classical twin", MATCHED, full),
    ("C→Q hybrid", CQ, full),
    ("Q→C hybrid", QC, full),
    ("Random filter + CNN", "Random patch + CNN (control)", full),
    ("Average pooling + CNN", "Avg-pool + CNN (control)", full),
    ("Learned filter + CNN", "Learned 2x2 conv + CNN (control)", full),
    ("ResNet18 + classical head", "ResNet18 + classical", comp),
    ("ResNet18 + C→Q head", "ResNet18 + quantum", comp),
]
rows = []
for label, key, df in METRIC_ROWS:
    r = df.loc[key]
    rows.append([label, f"{int(r.trainable_params):,}", f"{100 * r.accuracy_mean:.2f}",
                 f"{r.precision_mean:.3f}", f"{r.recall_mean:.3f}", f"{r.f1_mean:.3f}",
                 f"{r.roc_auc_mean:.3f}"])
table(s, L, T - Inches(0.05), CW, ["Model", "Parameters", "Accuracy (%)", "Precision", "Recall", "F1",
                                    "ROC-AUC"], rows,
      col_w=[3.6, 1.6, 1.65, 1.35, 1.35, 1.3, 1.4], size=18, highlight_rows=(2, 3, 8), row_h=0.5)
text(s, L, Inches(6.55), CW, Inches(0.4),
     "Average of 3 runs on the same 630 test images, full training data.  Highlighted = quantum models.",
     size=16, color=MUTED)

# ===========================================================================
# 13. Is it real?
# ===========================================================================
s = new_slide("Result 4: is the difference real?", notes=(
    "We used McNemar's test, which looks at the test images where two models disagree. A p-value "
    "below 0.05 means the difference is probably real. For the quantum head, the p-values were "
    "0.09, 0.15 and 1.0, so we cannot be sure the gain is real; the two models disagree on only a "
    "few images. The quantum filter is clearly better than the random filter, but not better than "
    "average pooling or the learned filter."))
table(s, L, T, CW, ["Comparison", "Run 1", "Run 2", "Run 3", "Real difference?"], [
    ["C→Q vs classical twin", "0.092", "0.146", "1.000", "No"],
    ["C→Q vs normal CNN", "0.424", "0.227", "0.267", "No"],
    ["Q→C vs random filter", "<0.001", "<0.001", "<0.001", "Yes"],
    ["Q→C vs average pooling", "1.000", "0.344", "0.791", "No"],
    ["Q→C vs learned filter", "1.000", "1.000", "0.057", "No"],
], col_w=[4.9, 1.55, 1.55, 1.55, 2.7], size=21, highlight_rows=(2,), row_h=0.62)
note_box(s, L, T + Inches(4.0), CW, Inches(1.15),
         "p-value below 0.05 means the difference is probably real.\n"
         "The small gain of C→Q could be due to chance.", size=22, color=ACCENT,
         fill=PALE_BLUE)

# ===========================================================================
# 14. Little data
# ===========================================================================
s = new_slide("Result 5: training with very little data", notes=(
    "We also trained with only 2 to 25 percent of the data. With just 58 images, the quantum filter "
    "model and average pooling did best, again pointing to image shrinking. The quantum head did "
    "badly on F1 at 2 percent, but it was the best at ranking images, with an AUC of 0.927; its "
    "problem is only the 0.5 cut-off. With 5 percent or more, the normal CNN was as good or better."))
picture(s, f"{SFIG}/lowdata_f1.png", L, T, Inches(7.6), CH)
bullets(s, Inches(8.35), T + Inches(0.2), Inches(4.55), CH, [
    "With 58 images, Q→C and average pooling do best",
    "C→Q ranks images best (AUC 0.927)",
    "- but its 0.5 cut-off is off",
    "With more data, the normal CNN is as good"], size=23, gap=18)

# ===========================================================================
# 15. Inside the circuit
# ===========================================================================
s = new_slide("Result 6: looking inside the quantum circuit", notes=(
    "We also studied the circuit itself. Left: with entanglement the circuit can produce a much "
    "richer set of quantum states, 20 to 60 times richer than without it. But the accuracy was the "
    "same, so our task does not need this extra power. Right: as we add qubits, the training signal, "
    "the gradient, gets about half as big for each extra qubit. This is called a barren plateau and "
    "it is why we kept the circuit small."))
picture(s, f"{FIG}/expressibility.png", L, T, Inches(6.0), Inches(4.1))
picture(s, f"{FIG}/barren_plateaus.png", Inches(6.85), T, Inches(6.0), Inches(4.1))
text(s, L, Inches(5.8), Inches(6.0), Inches(0.9),
     "Entanglement makes the circuit\nmuch richer, but accuracy stays the same",
     size=21, bold=True, align=PP_ALIGN.CENTER)
text(s, Inches(6.85), Inches(5.8), Inches(6.0), Inches(0.9),
     "Each extra qubit halves the gradient,\nso small circuits train best",
     size=21, bold=True, align=PP_ALIGN.CENTER)

# ===========================================================================
# 16. Noise
# ===========================================================================
s = new_slide("Result 7: it still works with noise", notes=(
    f"Real quantum computers are noisy and can only run a circuit a limited number of times. So we "
    f"tested our trained model with only 10 measurements per circuit, and with strong noise added. "
    f"The accuracy stayed within one test image of {exact:.2f} percent in every case. This means the "
    f"model should also work on today's quantum hardware."))
picture(s, f"{SFIG}/nisq_robustness.png", L, T, CW, Inches(4.45))
note_box(s, L, Inches(6.05), CW, Inches(0.7),
         "With only 10 measurements or strong noise, accuracy changes by at most 1 image.", size=22,
         color=ACCENT, fill=PALE_BLUE)

# ===========================================================================
# 17. Mistakes
# ===========================================================================
s = new_slide("Which X-rays did the model get wrong?", notes=(
    "These are the mistakes our CNN with a quantum head was most sure about. Most of its mistakes "
    "are missed TB cases, which is the more dangerous kind of mistake. One normal X-ray was called "
    "TB; it is a portable X-ray with text on it, which suggests the model may partly learn where an "
    "image came from, not only the disease."))
picture(s, f"{SFIG}/errors_labeled.png", L, T, CW, Inches(3.1))
bullets(s, L, T + Inches(3.4), CW, Inches(1.9), [
    "10 mistakes out of 630: 7 missed TB, 3 false alarms",
    "Only 22 images were wrong for any of our 4 main models",
    "Text and markers on X-rays may confuse the model"], size=24, gap=12)

# ===========================================================================
# 18. Conclusion and future
# ===========================================================================
s = new_slide("Conclusion and future work", notes=(
    "To sum up: a quantum classifier can replace a normal layer and gives a small gain, but we cannot "
    "prove it is real. The quantum filter's saving comes from shrinking the image. Small circuits "
    "train well and handle noise. Our main limits are that we only simulated the quantum computer, "
    "used small images and one dataset. Next, we would like to run the model on a real quantum "
    "computer, use bigger images, test on X-rays from other hospitals, and train the quantum filter."))
card(s, L, T, Inches(6.6), CH, "What we found", [
    "C→Q: small gain (+0.7%), not proven",
    "Q→C: saving comes from shrinking",
    "Small circuits train well",
    "Still works well with noise",
    "No clear quantum advantage yet"], size=24)
card(s, Inches(7.35), T, Inches(5.45), CH, "What comes next", [
    "Run on a real quantum computer",
    "Bigger images, X-rays from other hospitals",
    "Train the Q→C quantum filter",
    "Show which part of the X-ray the model uses"], color=QUANTUM, size=24)

# ===========================================================================
# 19. References
# ===========================================================================
s = new_slide("References", notes=(
    "These are our main references. The full list is in the report. The QR code opens our code."))
bullets(s, L, T, Inches(9.1), CH, [
    "WHO, Global Tuberculosis Report, 2024",
    "Rahman et al., TB detection using chest X-ray with deep learning, IEEE Access, 2020",
    "He et al., Deep residual learning for image recognition, CVPR, 2016",
    "Preskill, Quantum computing in the NISQ era and beyond, Quantum, 2018",
    "Mari et al., Transfer learning in hybrid classical-quantum neural networks, Quantum, 2020",
    "Henderson et al., Quanvolutional neural networks, Quantum Machine Intelligence, 2020",
    "McClean et al., Barren plateaus in quantum neural network training, Nature Comm., 2018",
    "Sim et al., Expressibility and entangling capability of PQCs, Adv. Quantum Tech., 2019"],
    size=18, gap=10)
picture(s, f"{SFIG}/repo_qr.png", Inches(9.95), T + Inches(0.2), Inches(2.9), Inches(2.9))
text(s, Inches(9.6), T + Inches(3.25), Inches(3.6), Inches(1.0), "Our code:\n" + REPO,
     size=15, color=MUTED, align=PP_ALIGN.CENTER)

# ===========================================================================
# 20. Thank you
# ===========================================================================
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
text(s, 0, Inches(1.2), W, Inches(1.2), "Thank you", size=50, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
text(s, 0, Inches(2.4), W, Inches(0.8), "Questions?", size=30, color=SOFT, align=PP_ALIGN.CENTER)
text(s, 0, Inches(3.8), W, Inches(2.0),
     "\n".join(f"{n}  ({reg})" for n, reg in STUDENTS), size=22, color=WHITE, align=PP_ALIGN.CENTER)
text(s, 0, Inches(5.9), W, Inches(0.6), f"Guide: {GUIDE}", size=22, color=SOFT, align=PP_ALIGN.CENTER)
s.notes_slide.notes_text_frame.text = (
    "Thank you for listening. We are happy to answer questions.\n\n"
    "Likely questions:\n"
    "Why 64 x 64 images? The quantum filter needs 1,024 circuits per image at 64 x 64; at 224 x 224 "
    "it would be about 12 times more, too slow on a laptop. All models use the same size, so it is fair.\n"
    "Why do the CNN numbers differ between slides 11 and 12? Slide 11 keeps the last training epoch; "
    "slide 12 keeps the epoch with the lowest validation loss. The quantum head got 98.41% in both.\n"
    "Why 2 circuit layers when 1 layer had the best validation score? 2 is our default in every "
    "experiment, and all depths are within one test image of each other.\n"
    "Could the model learn where the image came from? Yes, possibly: normal and TB images partly "
    "come from different hospitals and have different markers. This affects all models equally.\n"
    "Why McNemar per run? All models are tested on the same images, so a paired test fits, and each "
    "run is a different trained model.\n"
    "Why is the random filter so bad? Its tanh saturates and turns pixels almost black or white, "
    "so image detail is lost.")

# Slide numbers
total = len(prs.slides)
for i, sl in enumerate(prs.slides, 1):
    for shp in sl.shapes:
        if shp.has_text_frame and shp.text_frame.text == "" and shp.left > W - Inches(2):
            run = shp.text_frame.paragraphs[0].add_run()
            run.text = f"{i} / {total}"
            run.font.size, run.font.color.rgb, run.font.name = Pt(13), MUTED, FONT

os.makedirs(os.path.dirname(OUT), exist_ok=True)
prs.save(OUT)
print(f"saved {OUT} with {total} slides")

