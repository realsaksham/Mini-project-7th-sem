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


class ResNet18Backbone(nn.Module):
    """
    ImageNet-pretrained ResNet18 (transfer learning) -> Dense(embedding_dim).

    ResNet18 expects 3-channel input, so the single grayscale channel is
    repeated 3x. Its own 1000-way ImageNet classifier (`fc`) is dropped and
    replaced by a fresh Linear(512 -> embedding_dim), so it emits exactly the
    same-width embedding as CNNBackbone and either head can sit on top.

    The pretrained layers are fine-tuned at a reduced learning rate (see
    `pretrained_parameters()` and train.py's optimizer setup) so the
    ImageNet features are adapted rather than overwritten in the first steps.
    """

    def __init__(self, embedding_dim: int, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.body = resnet18(weights=weights)
        self.body.fc = nn.Identity()  # body now outputs the 512-d pooled features
        self.project = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)  # grayscale -> 3 identical channels
        return self.project(self.body(x))

    def pretrained_parameters(self):
        return self.body.parameters()

    @property
    def out_dim(self) -> int:
        return self.project.out_features


def _small_cnn(cfg) -> nn.Module:
    # With a patch-level input transform (quanvolution or one of its
    # classical controls) the CNN sees a half-resolution feature map instead
    # of the raw image: 4 channels, or 1 for plain average pooling.
    if cfg.input_transform == "none":
        size, channels = cfg.image_size, 1
    else:
        size, channels = cfg.image_size // 2, 1 if cfg.input_transform == "avgpool" else 4
    cnn = CNNBackbone(cfg.conv1_channels, cfg.conv2_channels, cfg.embedding_dim,
                      image_size=size, in_channels=channels)
    if cfg.input_transform == "learned_patch":
        # Trainable 1x1 conv over the 4 pixels of each 2x2 patch == a learned
        # classical 2x2-stride-2 filter; the classical twin of quanvolution.
        return nn.Sequential(nn.Conv2d(4, 4, kernel_size=1), nn.Tanh(), cnn)
    return cnn


# Registry pattern: each entry builds a backbone from a Config. Adding a new
# backbone is a single new entry here; nothing that reads BACKBONES changes.
BACKBONES = {
    # With a quanvolution / random-patch input transform the CNN sees a
    # 4-channel, half-resolution feature map instead of the raw image.
    "small_cnn": lambda cfg: _small_cnn(cfg),
    "resnet18": lambda cfg: ResNet18Backbone(cfg.embedding_dim, pretrained=True),
}


def build_backbone(cfg) -> nn.Module:
    if cfg.backbone_name not in BACKBONES:
        raise ValueError(f"Unknown backbone {cfg.backbone_name!r}; choose from {list(BACKBONES)}")
    return BACKBONES[cfg.backbone_name](cfg)


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


class MatchedClassicalHead(nn.Module):
    """
    The FAIR classical control for QuantumHead: identical structure with the
    quantum circuit swapped for a classical layer of (almost) the same size.

        Linear(embedding_dim -> 4) -> tanh*pi -> Linear(4 -> 4) -> tanh
        -> Linear(4 -> 1) -> Sigmoid

    The quantum layer maps 4 bounded angles to 4 values in [-1, 1] with
    depth*4*3 = 24 trainable weights (depth 2); Linear(4 -> 4) + tanh does the
    same with 20. Any accuracy gap between this head and QuantumHead is
    therefore down to *what* the 4->4 layer computes (quantum vs classical),
    not to extra width, parameters, or a different bottleneck.
    """

    def __init__(self, embedding_dim: int, width: int):
        super().__init__()
        self.pre = nn.Linear(embedding_dim, width)
        self.mid = nn.Linear(width, width)
        self.post = nn.Linear(width, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.tanh(self.pre(x)) * torch.pi
        x = torch.tanh(self.mid(x))
        return self.sigmoid(self.post(x))


# ---------------------------------------------------------------------------
# 3. Quantum head (hybrid classifier)
# ---------------------------------------------------------------------------


ENCODINGS = ("angle", "amplitude")
ENTANGLEMENTS = ("strong", "basic", "none")


def circuit_input_dim(n_qubits: int, encoding: str) -> int:
    """How many classical numbers the circuit's encoding step consumes."""
    return 2 ** n_qubits if encoding == "amplitude" else n_qubits


def circuit_weight_shape(n_qubits: int, depth: int, entanglement: str) -> tuple:
    """Shape of the trainable `weights` tensor for a given layer pattern."""
    if entanglement == "basic":
        return qml.BasicEntanglerLayers.shape(n_layers=depth, n_wires=n_qubits)
    # "strong" and "none" both use one 3-angle Rot per qubit per layer.
    return qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)


def make_quantum_circuit(n_qubits: int, depth: int,
                         encoding: str = "angle", entanglement: str = "strong"):
    """
    Build (and return) the PennyLane QNode that implements the variational
    quantum classifier, plus the weight_shapes dict qml.qnn.TorchLayer needs
    to allocate its trainable parameters.

    `encoding` and `entanglement` select the variants swept by src/ablation.py;
    the defaults ("angle", "strong") are the main hybrid model described below.

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
        #
        # Amplitude encoding (the alternative) instead writes 2**n_qubits
        # numbers directly into the statevector's amplitudes after scaling
        # them to unit length -- exponentially more features per qubit, but
        # the state-preparation circuit it implies is far deeper on hardware.
        if encoding == "amplitude":
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        else:
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
        #
        # Variants: "basic" uses BasicEntanglerLayers (a single RX per qubit
        # + nearest-neighbour CNOT ring, weights (depth, n_qubits)); "none"
        # applies the same per-qubit Rot gates but no CNOTs, so the qubits
        # never interact -- the control for "does entanglement matter?".
        if entanglement == "strong":
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        elif entanglement == "basic":
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        else:
            for layer in range(depth):
                for w in range(n_qubits):
                    qml.Rot(*weights[..., layer, w, :].unbind(-1), wires=w)

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
    weight_shapes = {"weights": circuit_weight_shape(n_qubits, depth, entanglement)}
    return circuit, weight_shapes


class QuantumHead(nn.Module):
    """
    Linear(embedding_dim -> n_qubits) -> tanh*pi (angle scaling) ->
    quantum circuit (TorchLayer) -> Linear(n_qubits -> 1) -> Sigmoid.

    The quantum layer replaces *only* the final dense classifier -- the
    output layer stays classical, as required.
    """

    def __init__(self, embedding_dim: int, n_qubits: int, depth: int,
                 encoding: str = "angle", entanglement: str = "strong"):
        super().__init__()
        if encoding not in ENCODINGS:
            raise ValueError(f"Unknown encoding {encoding!r}; choose from {ENCODINGS}")
        if entanglement not in ENTANGLEMENTS:
            raise ValueError(f"Unknown entanglement {entanglement!r}; choose from {ENTANGLEMENTS}")
        self.n_qubits = n_qubits
        self.depth = depth
        self.encoding = encoding
        self.input_dim = circuit_input_dim(n_qubits, encoding)

        # Classical pre-processing: compress the CNN embedding down to
        # exactly as many numbers as the encoding consumes (n_qubits for
        # angle encoding, 2**n_qubits for amplitude encoding).
        self.pre = nn.Linear(embedding_dim, self.input_dim)

        circuit, weight_shapes = make_quantum_circuit(n_qubits, depth, encoding, entanglement)
        self.circuit = circuit  # kept for qml.draw() / inspection in train.py
        self.weight_shapes = weight_shapes

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
        if self.encoding == "amplitude":
            # AmplitudeEmbedding normalizes the vector itself; no angle
            # range to respect, so the raw linear output goes straight in.
            return self.sigmoid(self.post(self.qlayer(x)))
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
        head_type: "classical" for the baseline, "quantum" for the hybrid,
            "classical_matched" for the parameter-matched classical control.
    """
    backbone = build_backbone(cfg)
    if head_type == "classical":
        head = ClassicalHead(cfg.embedding_dim)
    elif head_type == "classical_matched":
        head = MatchedClassicalHead(cfg.embedding_dim, cfg.n_qubits)
    elif head_type == "quantum":
        head = QuantumHead(cfg.embedding_dim, cfg.n_qubits, cfg.circuit_depth,
                           encoding=cfg.encoding, entanglement=cfg.entanglement)
    else:
        raise ValueError(f"Unknown head_type: {head_type!r}")
    return DiseaseClassifier(backbone, head)
