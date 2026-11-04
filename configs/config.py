"""
Single source of truth for every hyperparameter in this project.

Nothing in src/data.py, src/models.py, src/train.py, or src/ablation.py should
hardcode a hyperparameter value directly -- it should read it off a `Config`
instance instead. That way extending the experiments later (more qubits, a
different backbone, more epochs, ...) means editing this one file.
"""

from dataclasses import dataclass, field, replace


@dataclass
class Config:
    # --- reproducibility ---
    seed: int = 42                     # weight init + batch order
    # Seed for the TB train/val/test split, kept separate from `seed` so that
    # repeating a run with several seeds varies only initialization/shuffling
    # and every run is scored on the exact same test images.
    split_seed: int = 42

    # --- data ---
    dataset: str = "tb"                # key into DATASET_PRESETS below
    data_root: str = "./data"          # where medmnist caches its .npz download / TB cache lives
    image_size: int = 64               # images are resized to image_size x image_size grayscale
    class_names: list = field(default_factory=lambda: ["Normal", "Tuberculosis"])
    batch_size: int = 32
    # Used only when --subset is passed. Kept small so a full train/eval/plot
    # cycle finishes in well under a minute for debugging.
    subset_train_size: int = 200
    subset_val_size: int = 60
    subset_test_size: int = 60

    # --- TB Chest X-ray dataset (Kaggle: tawsifurrahman/tuberculosis-tb-chest-xray-dataset) ---
    # Folder containing the extracted Normal/ and Tuberculosis/ subfolders.
    tb_root: str = "./data/tb_raw/TB_Chest_Radiography_Database"
    # The Kaggle dataset ships without a split, so we make a stratified,
    # seeded one: 70% train / 15% val / 15% test.
    val_frac: float = 0.15
    test_frac: float = 0.15
    # Fraction of the training split actually used (stratified, seeded by
    # `seed`); <1.0 for the low-data experiments. Val/test are never reduced.
    train_fraction: float = 1.0

    # --- input transform (QUANTUM -> CLASSICAL hybrid), see src/quanvolution.py ---
    #   "none"         -- raw image into the CNN
    #   "quanv"        -- 2x2-patch quanvolution -> 4-channel quantum feature map
    #   "random_patch" -- classical control: fixed random tanh map on 2x2 patches
    #   "avgpool"      -- classical control: 2x2 average pooling (1 channel)
    #   "learned_patch"-- classical control: trainable 2x2 stride-2 conv + tanh
    input_transform: str = "none"
    quanv_layers: int = 1              # RandomLayers repetitions in the quanv filter
    quanv_seed: int = 0                # seed for the frozen random filter

    # --- backbone (classical CNN shared by baseline and hybrid) ---
    backbone_name: str = "small_cnn"   # key into models.BACKBONES ("small_cnn" | "resnet18")
    conv1_channels: int = 8
    conv2_channels: int = 16
    embedding_dim: int = 32            # output width of the backbone's final Dense layer

    # --- quantum head ---
    n_qubits: int = 4                  # also the input width of the quantum circuit
    circuit_depth: int = 2             # number of trainable-layer repetitions
    # How classical features are loaded into the circuit:
    #   "angle"     -- one feature per qubit as an RY rotation (n_qubits features)
    #   "amplitude" -- features become the statevector's amplitudes (2**n_qubits features)
    encoding: str = "angle"
    # Trainable-layer pattern:
    #   "strong" -- StronglyEntanglingLayers (Rot on every qubit + variable-range CNOT ring)
    #   "basic"  -- BasicEntanglerLayers (one RX per qubit + nearest-neighbour CNOT ring)
    #   "none"   -- Rot on every qubit, no CNOTs (control: no entanglement at all)
    entanglement: str = "strong"

    # --- training ---
    lr: float = 1e-3
    # Pretrained backbone layers (ResNet18) train at lr * this factor, so
    # fine-tuning adapts the ImageNet features instead of wiping them out.
    pretrained_lr_scale: float = 0.1
    epochs: int = 15
    # Keep the weights from the epoch with the lowest validation loss
    # (early-stopping style) instead of the last epoch's.
    restore_best: bool = False

    # --- results ---
    results_dir: str = "./results/tb"


# Per-dataset overrides applied on top of the Config defaults.
DATASET_PRESETS = {
    "tb": dict(
        image_size=64,
        class_names=["Normal", "Tuberculosis"],
        results_dir="./results/tb",
    ),
    "pneumoniamnist": dict(
        image_size=28,                 # PneumoniaMNIST is natively 28x28 grayscale
        class_names=["Normal", "Pneumonia"],
        results_dir="./results/pneumoniamnist",
    ),
}


def get_config(subset: bool = False, dataset: str = "tb") -> Config:
    """
    Return the project's default Config for `dataset`, or a fast-debug variant
    when `subset=True`: fewer epochs and a tiny slice of data so the whole
    pipeline (data -> model -> train -> eval -> save) can be sanity-checked
    in seconds before committing to a full run.
    """
    if dataset not in DATASET_PRESETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {list(DATASET_PRESETS)}")
    cfg = replace(Config(), dataset=dataset, **DATASET_PRESETS[dataset])
    if subset:
        cfg = replace(cfg, epochs=2, batch_size=16)
    return cfg
