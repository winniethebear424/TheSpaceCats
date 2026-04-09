"""
Handles all data transformations: per-channel min-max normalization (using training stats only), patch discretization, and random masking. 
Returns the masked image and a boolean mask indicating which patches were hidden.
"""

# CNN Encoder: (B, C, 64, 64) -> (B, latent_dim)

import numpy as np
import torch


class ChannelNormalizer:
    """Fit on train split only, then transform any split."""
    
    def __init__(self): 
        self.mins = self.maxs = None
        
    def fit(self, images: np.ndarray):
        # images shape: (N, C, H, W)
        self.mins = images.min(axis=(0,2,3), keepdims=True)
        self.maxs = images.max(axis=(0,2,3), keepdims=True)
        return self
        
    def transform(self, images: np.ndarray) -> np.ndarray:
        return (images - self.mins) / (self.maxs - self.mins + 1e-8)


def make_patch_mask(
    image_size: int,
    patch_size: int,
    mask_ratio: float,
    rng: np.random.Generator
) -> np.ndarray:
    """Returns bool mask (H, W): True = patch is masked (zeroed out)."""
    n = image_size // patch_size
    total = n * n
    n_masked = int(total * mask_ratio)
    idx = rng.permutation(total)[:n_masked]
    flat = np.zeros(total, dtype=bool)
    flat[idx] = True
    grid = flat.reshape(n, n)
    # Expand to pixel-level mask
    mask = np.repeat(np.repeat(grid, patch_size, axis=0), patch_size, axis=1)
    return mask  # (H, W)

def apply_mask(image: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """Zero out masked patches. image: (C, H, W)"""
    m = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
    return image.masked_fill(m, 0.0)