"""
Model definitions.

Layout, deliberately, so that training code (src/train.py) never has to know
*which* backbone or *which* classifier head it's training:

    backbone (CNN)  --embedding_dim-->  head (classical OR quantum)  --> prob.

    DiseaseClassifier = backbone + head           <- single nn.Module
    build_model(cfg, head_type)                   <- the only thing train.py calls

Swapping in a different backbone later (e.g. a pretrained ResNet18) means
adding one class + one BACKBONES entry below -- nothing in train.py changes.
Swapping n_qubits/circuit_depth means changing configs/config.py -- nothing
here changes either.
"""

import pennylane as qml
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1. Backbone registry
# ---------------------------------------------------------------------------


class CNNBackbone(nn.Module):
    """
    Conv-ReLU-MaxPool x2 -> Flatten -> Dense(embedding_dim).

    This is the feature extractor shared, unchanged, by both the baseline and
    the hybrid model -- the controlled variable in the comparison is only
    what happens to these `embedding_dim` features afterwards.
    """

    def __init__(self, conv1_channels: int, conv2_channels: int,
                 embedding_dim: int, image_size: int = 28, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, conv1_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # image_size -> image_size / 2

            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # image_size / 2 -> image_size / 4
        )
        flat_size = conv2_channels * (image_size // 4) * (image_size // 4)
        self.project = nn.Linear(flat_size, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.project(x)

    @property
    def out_dim(self) -> int:
        return self.project.out_features


# Registry pattern: adding a new backbone later (e.g. "resnet18") is a single
# new entry here. Nothing that reads BACKBONES needs to change.
BACKBONES = {
    "small_cnn": CNNBackbone,
}


def build_backbone(cfg) -> nn.Module:
    backbone_cls = BACKBONES[cfg.backbone_name]
    return backbone_cls(
        conv1_channels=cfg.conv1_channels,
        conv2_channels=cfg.conv2_channels,
        embedding_dim=cfg.embedding_dim,
        image_size=cfg.image_size,
    )


# ---------------------------------------------------------------------------
# 2. Classical head (baseline classifier)
# ---------------------------------------------------------------------------


class ClassicalHead(nn.Module):
    """Linear(embedding_dim -> 1) -> Sigmoid. The baseline's classifier."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))


# ---------------------------------------------------------------------------
# 3. Quantum head (hybrid classifier)
# ---------------------------------------------------------------------------


def make_quantum_circuit(n_qubits: int, depth: int):
    """
    Build (and return) the PennyLane QNode that implements the variational
    quantum classifier, plus the weight_shapes dict qml.qnn.TorchLayer needs
    to allocate its trainable parameters.

    Kept as a standalone function (rather than inlined in QuantumHead) so it
    can be constructed once, printed with qml.draw() for the report, and
    reused -- the same circuit object is wrapped by TorchLayer in QuantumHead.
    """
    # `default.qubit` is PennyLane's built-in statevector simulator. With only
    # n_qubits=4 wires, simulating the full 2^4 = 16-dimensional statevector
    # per sample is cheap -- this is what makes training on a laptop CPU
    # feasible at all.
    dev = qml.device("default.qubit", wires=n_qubits)

    # interface="torch" is what lets this QNode sit inside a PyTorch autograd
    # graph. diff_method="backprop" tells PennyLane to differentiate straight
    # through the simulator's linear-algebra ops (the way PyTorch would
    # differentiate any other tensor op), which is only valid in simulation --
    # it's the fast choice for training here. On real quantum hardware this
    # isn't possible (you can't backprop through a physical device), so a
    # deployed/hardware version of this circuit would instead use
    # diff_method="parameter-shift", which estimates each gradient component
    # by evaluating the circuit twice with that parameter shifted by +-pi/2.
    # That's exact but requires 2 circuit evaluations per trainable
    # parameter, per sample -- much slower, but hardware-compatible.
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # --- ENCODING: classical features -> quantum state (angle encoding) ---
        # `inputs` is a length-n_qubits vector of real numbers (already
        # squashed into [-pi, pi] by the tanh*pi step in QuantumHead, below).
        # AngleEmbedding maps inputs[i] -> a Pauli-Y rotation RY(inputs[i])
        # applied to qubit i, starting from the all-|0> state. A rotation by
        # angle theta about the Y-axis of the Bloch sphere turns |0> into
        # cos(theta/2)|0> + sin(theta/2)|1> -- i.e. the classical number is
        # encoded as *how far* that qubit's state has rotated, not stored as
        # an amplitude directly. This is "angle encoding": n_qubits classical
        # numbers in, n_qubits independently-rotated qubits out, one gate
        # each, no entanglement yet.
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')

        # --- TRAINABLE TRANSFORM: entangle the qubits ---
        # StronglyEntanglingLayers repeats, `depth` times:
        #   1. One arbitrary single-qubit rotation (qml.Rot, 3 trainable Euler
        #      angles) on every qubit -- this is where the learnable weights
        #      live.
        #   2. A ring of CNOT gates connecting each qubit to another qubit
        #      some distance away (the offset changes from layer to layer).
        #      CNOTs are what *entangle* the qubits -- after this step, the
        #      qubits' states are no longer independent, so the circuit can
        #      represent correlations between the 4 encoded features that a
        #      per-feature-independent operation never could.
        # weights has shape (depth, n_qubits, 3) -- see weight_shapes below --
        # so `depth` directly controls how many rounds of entanglement happen,
        # i.e. the circuit's expressive capacity. This is the knob the
        # ablation in src/ablation.py sweeps.
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        # --- MEASUREMENT: quantum state -> classical numbers ---
        # qml.expval(qml.PauliZ(w)) returns <psi|Z_w|psi>, the expectation
        # value of the Pauli-Z operator on wire w: +1 if that qubit is
        # definitely |0>, -1 if definitely |1>, and anything continuously in
        # between for a superposition -- always a real number in [-1, 1].
        # This is the standard bridge back to classical computation: it's the
        # only way to get a plain differentiable number out of a quantum
        # state (you can't read out the full statevector on real hardware,
        # and even in simulation we want a fixed-size output regardless of
        # n_qubits). We measure every wire, giving an n_qubits-length vector.
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    # StronglyEntanglingLayers.shape tells TorchLayer exactly how many
    # trainable parameters to allocate and in what shape, given `depth`
    # layers over `n_qubits` wires: (depth, n_qubits, 3).
    weight_shapes = {
        "weights": qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)
    }
    return circuit, weight_shapes


class QuantumHead(nn.Module):
    """
    Linear(embedding_dim -> n_qubits) -> tanh*pi (angle scaling) ->
    quantum circuit (TorchLayer) -> Linear(n_qubits -> 1) -> Sigmoid.

    The quantum layer replaces *only* the final dense classifier -- the
    output layer stays classical, as required.
    """

    def __init__(self, embedding_dim: int, n_qubits: int, depth: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth

        # Classical pre-processing: compress the CNN embedding down to
        # exactly n_qubits numbers, one per qubit.
        self.pre = nn.Linear(embedding_dim, n_qubits)

        circuit, weight_shapes = make_quantum_circuit(n_qubits, depth)
        self.circuit = circuit  # kept for qml.draw() / inspection in train.py

        # qml.qnn.TorchLayer turns the QNode into a torch.nn.Module: its
        # `weights` (shape (depth, n_qubits, 3)) become an nn.Parameter,
        # registered on this module exactly like a Linear layer's weight
        # matrix would be. Because of that, calling loss.backward() on the
        # full model's output walks gradients through this layer exactly like
        # any other -- autograd doesn't treat it specially, it just happens
        # to internally run a quantum circuit simulation forward and backward.
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Classical post-processing: the required classical output layer.
        self.post = nn.Linear(n_qubits, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pre(x)
        # Angle encoding needs inputs in a bounded range so rotation angles
        # don't wrap around past +-pi (which would make two different
        # classical feature values map to the same, or a confusingly
        # aliased, quantum rotation). tanh squashes to (-1, 1); multiplying
        # by pi stretches that to (-pi, pi) -- exactly one full, unambiguous
        # turn of the Bloch sphere per feature, and tanh's bounded derivative
        # also keeps this step numerically well-behaved during backprop.
        x = torch.tanh(x) * torch.pi
        x = self.qlayer(x)
        return self.sigmoid(self.post(x))


# ---------------------------------------------------------------------------
# 4. Full model + factory
# ---------------------------------------------------------------------------


class DiseaseClassifier(nn.Module):
    """backbone -> head -> probability. The only model class train.py sees."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def build_model(cfg, head_type: str) -> DiseaseClassifier:
    """
    The single factory function training code calls.

    Args:
        cfg: a configs.config.Config instance.
        head_type: "classical" for the baseline, "quantum" for the hybrid.
    """
    backbone = build_backbone(cfg)
    if head_type == "classical":
        head = ClassicalHead(cfg.embedding_dim)
    elif head_type == "quantum":
        head = QuantumHead(cfg.embedding_dim, cfg.n_qubits, cfg.circuit_depth)
    else:
        raise ValueError(f"Unknown head_type: {head_type!r}")
    return DiseaseClassifier(backbone, head)
