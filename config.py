# Central source of truth for all settings.
# Import this in every other module: from config import CFG

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    # --- Reproducibility ---
    seed: int = 42

    # --- Paths ---
    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = Path("checkpoints")

    # --- Dataset ---
    bands: list = field(default_factory=lambda: ["g", "r", "i", "z", "y"])
    image_size: int = 64
    train_frac: float = 0.7
    val_frac: float = 0.15
    # test_frac is implicitly 1 - train_frac - val_frac

    # --- Masking ---
    patch_size: int = 8   # 64/8 = 8x8 grid of patches
    mask_ratio: float = 0.0 # 0.0 = baseline, try 0.25/0.5/0.75

    # --- Model ---
    latent_dim: int = 128
    hidden_dims: list = field(default_factory=lambda: [32, 64, 128])
    kernel_size: int = 3

    # --- Training ---
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    loss_fn: str = "smooth_l1" # l1 | l2 | smooth_l1 | ssim | combo

CFG = Config()