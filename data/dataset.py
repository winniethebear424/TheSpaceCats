"""
PyTorch Dataset and DataLoader factory. Reads HDF5 galaxy images, applies normalization and masking on-the-fly, and returns 
(masked_input, original_image, mask) tuples for training.
"""


import h5py, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from config import CFG
from data.preprocess import ChannelNormalizer, make_patch_mask, apply_mask

class GalaxyDataset(Dataset):
    def __init__(self, h5_path, indices, normalizer,
                 mask_ratio=0.0, patch_size=8, seed=42):
        self.h5_path   = h5_path
        self.indices   = indices
        self.normalizer = normalizer
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)
        self.hf = None  # open lazily (HDF5 + multiprocessing)


    def __len__(self): return len(self.indices)

    def _open(self):
        if self.hf is None:
            self.hf = h5py.File(self.h5_path, "r")

    def __getitem__(self, idx):
        self._open()
        raw = self.hf["images"][self.indices[idx]]  # (C, H, W)
        img = self.normalizer.transform(raw[np.newaxis])[0]
        tensor = torch.tensor(img, dtype=torch.float32)
        mask = make_patch_mask(
            CFG.image_size, self.patch_size, self.mask_ratio, self.rng
        )
        masked = apply_mask(tensor, mask)
        return masked, tensor, torch.from_numpy(mask)


def make_dataloaders(h5_path, normalizer, mask_ratio, seed=42):
    n = ... # read total count from HDF5
    idx = np.random.default_rng(seed).permutation(n)
    t = int(n * CFG.train_frac); v = t + int(n * CFG.val_frac)
    splits = {"train": idx[:t], "val": idx[t:v], "test": idx[v:]}
    loaders = {}
    for split, ids in splits.items():
        ds = GalaxyDataset(h5_path, ids, normalizer, mask_ratio)
        loaders[split] = DataLoader(
            ds, batch_size=CFG.batch_size,
            shuffle=(split == "train"), num_workers=4
        )
    return loaders