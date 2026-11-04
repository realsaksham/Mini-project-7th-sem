"""
Quanvolution: a quantum circuit used as a convolution filter on the image
(Henderson et al., "Quanvolutional Neural Networks", 2019). This is the
QUANTUM -> CLASSICAL hybrid: quantum processing happens first, directly on
pixels, and a classical CNN learns from its output.

    64x64 X-ray  --2x2 patches, stride 2-->  4 pixels per patch
                 --4-qubit circuit-->        4 expectation values per patch
                 => 4-channel 32x32 "quantum feature map"  --> classical CNN

For each 2x2 patch:
  1. ENCODE: pixel p in [0, 1] -> RY(pi * p) on its own qubit (angle encoding).
  2. TRANSFORM: a fixed, randomly drawn circuit (qml.RandomLayers: random
     single-qubit rotations + CNOTs) mixes and entangles the 4 qubits.
  3. MEASURE: <Z> on each qubit -> 4 numbers in [-1, 1], one per channel.

The circuit's parameters are random and frozen (as in the original paper):
that makes the quantum transform a fixed feature map that can be computed
once for the whole dataset and cached, instead of re-simulating ~1000
circuits per image every epoch. Only the classical CNN after it is trained.

`random_patch_features` is the matching CLASSICAL control -- a fixed random
non-linear map with the same 4-in/4-out, bounded-output shape -- so the
comparison can tell "the quantum transform helps" apart from "any fixed
non-linear 2x2 patch transform helps".
"""

import os

import numpy as np
import pennylane as qml
import torch

N_QUBITS = 4   # one qubit per pixel of a 2x2 patch
PATCH = 2


def make_quanv_circuit(n_layers: int, seed: int):
    """Return (qnode, frozen_weights) for the quanvolution filter."""
    dev = qml.device("default.qubit", wires=N_QUBITS)
    rng = np.random.default_rng(seed)
    weights = torch.tensor(rng.uniform(0, 2 * np.pi, size=(n_layers, N_QUBITS)))

    @qml.qnode(dev, interface="torch")
    def circuit(patch_angles, weights):
        # patch_angles: (batch, 4) -- PennyLane broadcasts over the batch
        # axis, simulating every patch in one vectorized call.
        qml.AngleEmbedding(patch_angles, wires=range(N_QUBITS), rotation="Y")
        qml.RandomLayers(weights, wires=range(N_QUBITS), seed=seed)
        return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]

    return circuit, weights


def _patches(images: np.ndarray) -> np.ndarray:
    """(N, H, W) uint8 -> (N, H/2, W/2, 4) floats in [0, 1], one row per 2x2 patch."""
    n, h, w = images.shape
    x = images.astype(np.float32) / 255.0
    x = x.reshape(n, h // PATCH, PATCH, w // PATCH, PATCH).transpose(0, 1, 3, 2, 4)
    return x.reshape(n, h // PATCH, w // PATCH, PATCH * PATCH)


def quanv_features(images: np.ndarray, n_layers: int = 1, seed: int = 0,
                   chunk: int = 16) -> np.ndarray:
    """(N, H, W) uint8 -> (N, 4, H/2, W/2) float32 quantum feature maps."""
    circuit, weights = make_quanv_circuit(n_layers, seed)
    patches = _patches(images)
    n, ph, pw, _ = patches.shape
    out = np.empty((n, N_QUBITS, ph, pw), dtype=np.float32)
    for start in range(0, n, chunk):
        block = patches[start:start + chunk].reshape(-1, N_QUBITS)
        angles = torch.tensor(np.pi * block, dtype=torch.float64)
        with torch.no_grad():
            z = torch.stack(circuit(angles, weights), dim=-1).numpy()   # (b*ph*pw, 4)
        out[start:start + chunk] = z.reshape(-1, ph, pw, N_QUBITS).transpose(0, 3, 1, 2)
        if start // chunk % 50 == 0:
            print(f"  quanvolution: {min(start + chunk, n)}/{n} images")
    return out


def random_patch_features(images: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Classical control: tanh(W @ (pi * patch) + b) with a fixed random W, b.
    Same input (4 pixel angles), same output shape (4 values in [-1, 1]).
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1, size=(N_QUBITS, N_QUBITS)).astype(np.float32)
    b = rng.uniform(-np.pi, np.pi, size=N_QUBITS).astype(np.float32)
    z = np.tanh((np.pi * _patches(images)) @ W.T + b)          # (N, ph, pw, 4)
    return z.transpose(0, 3, 1, 2).astype(np.float32)


def avgpool_features(images: np.ndarray) -> np.ndarray:
    """Classical control: 2x2 average pooling -> (N, 1, H/2, W/2) in [-1, 1]."""
    return (_patches(images).mean(axis=-1) * 2 - 1)[:, None].astype(np.float32)


def patch_pixels(images: np.ndarray) -> np.ndarray:
    """
    Raw 2x2 patches as 4 channels -> (N, 4, H/2, W/2) in [-1, 1]. Used by the
    "learned_patch" control, where a trainable 1x1 conv + tanh on these
    channels is exactly a trainable 2x2-stride-2 classical conv filter.
    """
    return (_patches(images) * 2 - 1).transpose(0, 3, 1, 2).astype(np.float32)


def cached_features(kind: str, images: np.ndarray, cfg) -> np.ndarray:
    """Compute (or load from data_root) the transformed dataset for `kind`."""
    if kind in ("avgpool", "learned_patch"):
        return avgpool_features(images) if kind == "avgpool" else patch_pixels(images)
    if kind == "quanv":
        name = f"quanv_{cfg.image_size}_L{cfg.quanv_layers}_s{cfg.quanv_seed}.npz"
    else:
        name = f"randpatch_{cfg.image_size}_s{cfg.quanv_seed}.npz"
    path = os.path.join(cfg.data_root, name)
    if os.path.exists(path):
        return np.load(path)["features"]
    print(f"[data] computing {kind} features for {len(images)} images (one-time; cached to {path})")
    if kind == "quanv":
        feats = quanv_features(images, cfg.quanv_layers, cfg.quanv_seed)
    else:
        feats = random_patch_features(images, cfg.quanv_seed)
    np.savez_compressed(path, features=feats)
    return feats
