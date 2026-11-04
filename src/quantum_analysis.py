"""
Quantum circuit analysis -- properties of the circuits themselves, plus how
the trained C->Q hybrid behaves on (simulated) realistic quantum hardware.

  1. Expressibility (Sim, Johnson & Aspuru-Guzik, 2019): how uniformly the
     ansatz covers the space of quantum states. Measured as the KL divergence
     between the distribution of fidelities |<psi(a)|psi(b)>|^2 of random
     parameter pairs and that of Haar-random states,
         P_Haar(F) = (N - 1)(1 - F)^(N - 2),  N = 2^n_qubits.
     LOWER = more expressive.
  2. Entangling capability: average Meyer-Wallach measure
         Q = 2 (1 - mean_k Tr(rho_k^2))
     over random parameters (0 = product states only, 1 = maximal).
  3. Barren plateaus (McClean et al., 2018; Cerezo et al., 2021): variance of
     a cost gradient over random initializations vs. qubit count. Exponential
     decay means the circuit becomes untrainable as it grows; a LOCAL cost
     (<Z_0>) decays far more slowly than a GLOBAL one (<Z x Z x ... x Z>).
  4. Hardware realism for the trained C->Q model: test-set performance with
     (a) finite measurement shots and (b) depolarizing noise after every
     layer (density-matrix simulation).
  5. Gradient check: parameter-shift rule (hardware-compatible) vs. backprop
     (simulator-only) on the trained circuit -- they must agree.

Usage:
    python -m src.quantum_analysis           # full analysis
    python -m src.quantum_analysis --quick   # fewer samples, for debugging
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch

from configs.config import get_config
from src.data import load_data, set_seed
from src.evaluate import compute_metrics
from src.models import circuit_weight_shape, make_quantum_circuit, build_model
from src.quanvolution import make_quanv_circuit
from src.train import train_model

ANSATZE = ("strong", "basic", "none")
COLORS = {"strong": "#2a78d6", "basic": "#eb6834", "none": "#1baf7a"}
MARKERS = {"strong": "o", "basic": "s", "none": "D"}
INK, MUTED_INK = "#0b0b0b", "#52514e"


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=MUTED_INK)
    ax.set_ylabel(ylabel, color=MUTED_INK)
    ax.set_title(title, loc="left", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED_INK)
    ax.grid(True, alpha=0.25)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _apply_ansatz(weights, n_qubits, depth, entanglement):
    """The trainable block of the C->Q head, applied on its own."""
    if entanglement == "strong":
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    elif entanglement == "basic":
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    else:
        for layer in range(depth):
            for w in range(n_qubits):
                qml.Rot(*weights[..., layer, w, :].unbind(-1), wires=w)


# ---------------------------------------------------------------------------
# 1 + 2. Expressibility and entangling capability
# ---------------------------------------------------------------------------


def random_states(n_qubits, depth, entanglement, n_samples, rng):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def state(weights):
        _apply_ansatz(weights, n_qubits, depth, entanglement)
        return qml.state()

    shape = circuit_weight_shape(n_qubits, depth, entanglement)
    w = torch.tensor(rng.uniform(0, 2 * np.pi, size=(n_samples, *shape)))
    return state(w).numpy()  # (n_samples, 2^n)


def expressibility(states_a, states_b, n_bins=75):
    n = states_a.shape[1]
    fid = np.abs(np.sum(states_a.conj() * states_b, axis=1)) ** 2
    edges = np.linspace(0, 1, n_bins + 1)
    p_circ, _ = np.histogram(fid, bins=edges)
    p_circ = p_circ / p_circ.sum()
    # Haar probability mass per bin: integral of (N-1)(1-F)^(N-2) = -(1-F)^(N-1)
    p_haar = (1 - edges[:-1]) ** (n - 1) - (1 - edges[1:]) ** (n - 1)
    mask = p_circ > 0
    return float(np.sum(p_circ[mask] * np.log(p_circ[mask] / p_haar[mask])))


def meyer_wallach(states, n_qubits):
    psi = states.reshape(len(states), *([2] * n_qubits))
    purities = []
    for k in range(n_qubits):
        m = np.moveaxis(psi, k + 1, 1).reshape(len(states), 2, -1)
        rho = m @ m.conj().transpose(0, 2, 1)              # (S, 2, 2) reduced state of qubit k
        purities.append(np.real(np.einsum("sij,sji->s", rho, rho)))
    return float(np.mean(2 * (1 - np.mean(purities, axis=0))))


def run_expressibility(out_dir, n_samples, n_qubits=4, depths=(1, 2, 3, 4)):
    rng = np.random.default_rng(0)
    rows = []
    for ent in ANSATZE:
        for depth in depths:
            a = random_states(n_qubits, depth, ent, n_samples, rng)
            b = random_states(n_qubits, depth, ent, n_samples, rng)
            rows.append({"entanglement": ent, "depth": depth,
                         "expressibility_kl": expressibility(a, b),
                         "entangling_capability": meyer_wallach(np.concatenate([a, b]), n_qubits)})
            print(f"  {ent:<6} depth={depth}: expr={rows[-1]['expressibility_kl']:.4f} "
                  f"ent_cap={rows[-1]['entangling_capability']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "expressibility.csv"), index=False)

    for col, ylabel, title, name, log in [
        ("expressibility_kl", "KL divergence to Haar (lower = more expressive)",
         "Expressibility vs. depth (4 qubits)", "expressibility.png", True),
        ("entangling_capability", "Meyer-Wallach Q (0 = no entanglement)",
         "Entangling capability vs. depth (4 qubits)", "entangling_capability.png", False),
    ]:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for ent, g in df.groupby("entanglement", sort=False):
            ax.plot(g["depth"], g[col], lw=2, ms=8, color=COLORS[ent], marker=MARKERS[ent],
                    label=f"{ent} entanglement")
        if log:
            ax.set_yscale("log")
        ax.set_xticks(list(depths))
        ax.legend(frameon=False, labelcolor=INK)
        _style(ax, "Circuit depth (layers)", ylabel, title)
        _save(fig, os.path.join(out_dir, name))
    return df


# ---------------------------------------------------------------------------
# 3. Barren plateaus
# ---------------------------------------------------------------------------


def grad_variance(n_qubits, depth, cost, n_samples, rng):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(weights):
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        if cost == "local":
            return qml.expval(qml.PauliZ(0))
        return qml.expval(qml.prod(*[qml.PauliZ(w) for w in range(n_qubits)]))

    shape = qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)
    w = torch.tensor(rng.uniform(0, 2 * np.pi, size=(n_samples, *shape)), requires_grad=True)
    circuit(w).sum().backward()          # samples are independent -> per-sample grads
    g = w.grad[:, 0, 0, 1].numpy()       # d cost / d (first qubit's RY angle, layer 1)
    return float(np.var(g))


def run_barren_plateaus(out_dir, n_samples, qubits=(2, 4, 6, 8, 10), depths=(2, 10)):
    rng = np.random.default_rng(1)
    rows = []
    for cost in ("local", "global"):
        for depth in depths:
            for n in qubits:
                v = grad_variance(n, depth, cost, n_samples, rng)
                rows.append({"cost": cost, "depth": depth, "n_qubits": n, "grad_variance": v})
                print(f"  {cost:<6} depth={depth:<2} qubits={n:<2}: Var[dC/dtheta]={v:.3e}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "barren_plateaus.csv"), index=False)

    styles = {("local", depths[0]): dict(color="#2a78d6", ls="-", marker="o"),
              ("local", depths[1]): dict(color="#2a78d6", ls="--", marker="s"),
              ("global", depths[0]): dict(color="#eb6834", ls="-", marker="o"),
              ("global", depths[1]): dict(color="#eb6834", ls="--", marker="s")}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for (cost, depth), g in df.groupby(["cost", "depth"]):
        ax.plot(g["n_qubits"], g["grad_variance"], lw=2, ms=8,
                label=f"{cost} cost, depth {depth}", **styles[(cost, depth)])
    ax.set_yscale("log")
    ax.set_xticks(list(qubits))
    ax.legend(frameon=False, labelcolor=INK)
    _style(ax, "Number of qubits", "Var[∂C/∂θ] over random inits (log)",
           "Barren plateaus: gradient variance vs. qubits")
    _save(fig, os.path.join(out_dir, "barren_plateaus.png"))
    return df


# ---------------------------------------------------------------------------
# 4 + 5. Trained C->Q model: shots, noise, gradient check
# ---------------------------------------------------------------------------


def noisy_head_circuit(n_qubits, depth, noise_p=0.0, shots=None):
    """The trained head's circuit on a noisy / finite-shot device."""
    if noise_p > 0:
        dev = qml.device("default.mixed", wires=n_qubits)
    else:
        dev = qml.device("default.qubit", wires=n_qubits)
    ranges = [(l % (n_qubits - 1)) + 1 for l in range(depth)]  # StronglyEntanglingLayers' default

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for l in range(depth):
            # One layer at a time (same CNOT ranges as the full template), so
            # noise can be inserted after every layer like on real hardware.
            qml.StronglyEntanglingLayers(weights[l:l + 1], wires=range(n_qubits), ranges=[ranges[l]])
            if noise_p > 0:
                for w in range(n_qubits):
                    qml.DepolarizingChannel(noise_p, wires=w)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    if shots is not None:
        circuit = qml.set_shots(circuit, shots=shots)
    return circuit


@torch.no_grad()
def eval_with_circuit(model, circuit, loader, chunk=64):
    """Run the trained model with its quantum layer replaced by `circuit`."""
    head = model.head
    weights = head.qlayer.weights.detach().double()
    probs, labels = [], []
    for images, y in loader:
        angles = torch.tanh(head.pre(model.backbone(images))) * torch.pi
        out = []
        for i in range(0, len(angles), chunk):
            out.append(torch.stack(circuit(angles[i:i + chunk].double(), weights), dim=-1))
        z = torch.cat(out).float()
        probs.append(head.sigmoid(head.post(z)).squeeze(1))
        labels.append(y.squeeze(1).float())
    return compute_metrics(torch.cat(probs).numpy(), torch.cat(labels).numpy())[0]


def run_hardware_realism(out_dir, quick):
    cfg = get_config()
    set_seed(cfg.seed)
    train_loader, val_loader, test_loader = load_data(cfg)
    if quick:
        from src.data import _make_subset
        test_loader = torch.utils.data.DataLoader(
            _make_subset(test_loader.dataset, 128, cfg.seed), batch_size=cfg.batch_size)
    print("  training the default C->Q hybrid (seed 42) to analyse ...")
    model = build_model(cfg, head_type="quantum")
    train_model(model, train_loader, val_loader, cfg)
    model.eval()

    n, depth = cfg.n_qubits, cfg.circuit_depth
    exact = eval_with_circuit(model, noisy_head_circuit(n, depth), test_loader)
    print(f"  exact simulator: acc={exact['accuracy']:.4f} auc={exact['roc_auc']:.4f}")

    rows = [{"setting": "exact", "shots": None, "noise_p": 0.0, **exact}]
    for shots in ((10, 100, 1000) if quick else (10, 50, 100, 500, 1000, 10000)):
        torch.manual_seed(0)
        m = eval_with_circuit(model, noisy_head_circuit(n, depth, shots=shots), test_loader)
        rows.append({"setting": "shots", "shots": shots, "noise_p": 0.0, **m})
        print(f"  shots={shots:<6}: acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f}")
    for p in ((0.01, 0.1) if quick else (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)):
        m = eval_with_circuit(model, noisy_head_circuit(n, depth, noise_p=p), test_loader)
        rows.append({"setting": "depolarizing", "shots": None, "noise_p": p, **m})
        print(f"  depolarizing p={p:<5}: acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "hardware_realism.csv"), index=False)

    for setting, xcol, xlabel, title, name in [
        ("shots", "shots", "Measurement shots per expectation value (log)",
         "Trained C->Q model vs. measurement shots", "shots_robustness.png"),
        ("depolarizing", "noise_p", "Depolarizing probability per qubit per layer (log)",
         "Trained C->Q model vs. hardware noise", "noise_robustness.png"),
    ]:
        g = df[df["setting"] == setting]
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(g[xcol], g["accuracy"], lw=2, ms=8, marker="o", color="#2a78d6", label="accuracy")
        ax.plot(g[xcol], g["f1"], lw=2, ms=8, marker="s", color="#eb6834", label="F1")
        ax.axhline(exact["accuracy"], color="#2a78d6", lw=1, ls=":", label="accuracy, noiseless")
        ax.axhline(exact["f1"], color="#eb6834", lw=1, ls=":", label="F1, noiseless")
        ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        _style(ax, xlabel, "Test metric", title)
        _save(fig, os.path.join(out_dir, name))

    # --- gradient check: parameter-shift vs. backprop on the trained circuit ---
    images, _ = next(iter(test_loader))
    with torch.no_grad():
        angles = (torch.tanh(model.head.pre(model.backbone(images[:4]))) * torch.pi).double()
    grads = {}
    for method in ("backprop", "parameter-shift"):
        dev = qml.device("default.qubit", wires=n)
        circuit, _ = make_quantum_circuit(n, depth)
        qnode = qml.QNode(circuit.func, dev, interface="torch", diff_method=method)
        w = model.head.qlayer.weights.detach().double().clone().requires_grad_(True)
        total = sum(torch.stack(qnode(a, w)).sum() for a in angles)
        total.backward()
        grads[method] = w.grad.numpy()
    diff = float(np.max(np.abs(grads["backprop"] - grads["parameter-shift"])))
    return df, exact, diff


# ---------------------------------------------------------------------------


def circuit_diagrams(out_dir):
    cfg = get_config()
    text = []
    for ent in ANSATZE:
        circuit, shapes = make_quantum_circuit(cfg.n_qubits, cfg.circuit_depth, "angle", ent)
        d = qml.draw(circuit, level="device")(torch.ones(cfg.n_qubits), torch.zeros(shapes["weights"]))
        text += [f"=== C->Q head circuit: angle encoding, {ent} entanglement, depth {cfg.circuit_depth} ===", d, ""]
    circuit, shapes = make_quantum_circuit(cfg.n_qubits, cfg.circuit_depth, "amplitude", "strong")
    d = qml.draw(circuit)(torch.ones(2 ** cfg.n_qubits), torch.zeros(shapes["weights"]))
    text += ["=== C->Q head circuit: amplitude encoding (16 features -> 4 qubits) ===", d, ""]
    qc, w = make_quanv_circuit(cfg.quanv_layers, cfg.quanv_seed)
    d = qml.draw(qc, level="device")(torch.full((4,), 0.5), w)
    text += ["=== Q->C quanvolution filter (one 2x2 patch -> 4 channels) ===", d, ""]
    path = os.path.join(out_dir, "circuits.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Quantum circuit analysis.")
    parser.add_argument("--quick", action="store_true", help="Fewer samples, for debugging.")
    args = parser.parse_args()
    out_dir = os.path.join(get_config().results_dir, "quantum_analysis")
    os.makedirs(out_dir, exist_ok=True)

    circuit_diagrams(out_dir)
    print("\n[1-2] expressibility + entangling capability")
    expr = run_expressibility(out_dir, n_samples=500 if args.quick else 5000)
    print("\n[3] barren plateaus")
    bp = run_barren_plateaus(out_dir, n_samples=50 if args.quick else 500)
    print("\n[4-5] trained C->Q model on noisy / finite-shot hardware + gradient check")
    hw, exact, grad_diff = run_hardware_realism(out_dir, args.quick)

    md = ["# Quantum circuit analysis", "",
          "## Expressibility and entangling capability (4 qubits)", "",
          expr.to_markdown(index=False, floatfmt=".4f"), "",
          "## Barren plateaus (StronglyEntanglingLayers, random init)", "",
          bp.pivot_table(index="n_qubits", columns=["cost", "depth"], values="grad_variance")
            .to_markdown(floatfmt=".3e"), "",
          "## Trained C->Q model on realistic hardware (test set)", "",
          hw[["setting", "shots", "noise_p", "accuracy", "recall", "f1", "roc_auc"]]
            .to_markdown(index=False, floatfmt=".4f"), "",
          "## Gradient check", "",
          f"Max |backprop - parameter-shift| over all circuit weights: {grad_diff:.2e} "
          "(parameter-shift is what real hardware would use).", ""]
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Saved {path}")
    print(f"gradient check max diff: {grad_diff:.2e}")


if __name__ == "__main__":
    sys.exit(main())
