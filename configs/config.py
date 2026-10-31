"""
Single source of truth for every hyperparameter in this project.

Nothing in src/data.py, src/models.py, src/train.py, or src/ablation.py should
hardcode a hyperparameter value directly -- it should read it off a `Config`
instance instead. That way extending the experiments later (more qubits, a
different backbone, more epochs, ...) means editing this one file.
"""

from dataclasses import dataclass, replace


@dataclass
class Config:
    # --- reproducibility ---
    seed: int = 42

    # --- data ---
    data_root: str = "./data"          # where medmnist caches its .npz download
    image_size: int = 28               # PneumoniaMNIST is natively 28x28 grayscale
    batch_size: int = 32
    # Used only when --subset is passed. Kept small so a full train/eval/plot
    # cycle finishes in well under a minute for debugging.
    subset_train_size: int = 200
    subset_val_size: int = 60
    subset_test_size: int = 60

    # --- backbone (classical CNN shared by baseline and hybrid) ---
    backbone_name: str = "small_cnn"   # key into models.BACKBONES
    conv1_channels: int = 8
    conv2_channels: int = 16
    embedding_dim: int = 32            # output width of the backbone's final Dense layer

    # --- quantum head ---
    n_qubits: int = 4                  # also the input width of the quantum circuit
    circuit_depth: int = 2             # number of StronglyEntanglingLayers repetitions

    # --- training ---
    lr: float = 1e-3
    epochs: int = 15

    # --- results ---
    results_dir: str = "./results"


def get_config(subset: bool = False) -> Config:
    """
    Return the project's default Config, or a fast-debug variant when
    `subset=True`: fewer epochs and a tiny slice of data so the whole
    pipeline (data -> model -> train -> eval -> save) can be sanity-checked
    in seconds before committing to a full run.
    """
    cfg = Config()
    if subset:
        cfg = replace(cfg, epochs=2, batch_size=16)
    return cfg
