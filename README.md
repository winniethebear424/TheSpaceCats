

# Galaxy Masked Autoencoder — The Space Cats

## What this project does
This project trains a CNN-based masked autoencoder (CNN-AE) on galaxy images from the GalaxiesML dataset. The core idea is to train the model to reconstruct only the randomly masked patches of each image, forcing the encoder to learn richer global structure rather than memorizing pixel-level details. We compare a baseline (no masking) against different mask ratios.


## Setup

```bash
pip install torch torchvision torchmetrics h5py numpy scikit-learn umap-learn tqdm
```

Before running anything, inspect the HDF5 file to confirm the dataset key:

```python
import h5py
with h5py.File('path/to/data.h5', 'r') as f:
    print(list(f.keys()))
```

---

## Project structure

```
project/
├── config.py              # all hyperparameters and paths (start here)
├── train.py               # training loop
├── evaluate.py            # test-set evaluation + latent extraction
├── data/
│   ├── preprocess.py      # normalization + masking
│   └── dataset.py         # PyTorch Dataset and DataLoaders
├── models/
│   ├── encoder.py         # CNN encoder
│   ├── decoder.py         # CNN decoder
│   └── autoencoder.py     # wrapper model
└── utils/
    └── losses.py          # masked loss function builder
```

---

## Running an experiment
Each teammate owns one `mask_ratio` ablation. **Do not change the dataset split, normalization, or architecture** — only your `mask_ratio` and tuning hyperparameters should differ.

```bash
# Step 1 — set your mask_ratio in config.py (or pass via argparse once added)
# Step 2 — train
python train.py

# Step 3 — evaluate after training finishes
python evaluate.py

# Step 4 — share your outputs/ folder with the team
```

---

## File reference

### `config.py` — single source of truth for all settings

**What it does:**
- Defines a `Config` dataclass (`CFG`) holding every hyperparameter and path
- Controls seed, data paths, band list, image size, `patch_size`, `mask_ratio`
- Controls model capacity: `latent_dim`, `hidden_dims`, `kernel_size`
- Controls training: `batch_size`, `epochs`, `learning_rate`, `weight_decay`, `loss_fn`
- All other files import `CFG` — no magic numbers anywhere else

**TODO:**
- [ ] Agree on final `hidden_dims` and `latent_dim` values as a team — everyone must use the same
- [ ] Confirm `patch_size=8` gives a clean 8×8 patch grid on 64×64 images
- [ ] Add `ssim_weight` field for the combo loss (currently hardcoded in `losses.py`)
- [ ] Add argparse to `train.py` so `mask_ratio` can be overridden at the command line without editing this file

> **Note:** Each teammate tunes their own ablation by overriding `CFG.mask_ratio` — do not commit personal experiment values to `config.py`.

---

### `data/preprocess.py` — normalization and patch masking

**What it does:**
- `ChannelNormalizer`: fits per-channel min-max stats on the training split only, then transforms any split
- `make_patch_mask()`: randomly selects `mask_ratio` of the patches, returns a boolean pixel-level mask (`True` = masked)
- `apply_mask()`: zeros out masked patches on a `(C, H, W)` tensor
- Returns the mask separately so the loss function knows which patches to penalize

**TODO:**
- [ ] Investigate outlier pixel values in raw HDF5 data — consider percentile clipping (e.g. 1%–99%) instead of global min-max, which is sensitive to cosmic ray spikes
- [ ] Save fitted normalizer stats (`mins`/`maxs`) to disk so all teammates use identical normalization
- [ ] Write a quick sanity-check: plot one original vs. masked image for each `mask_ratio`

> **Note:** The normalizer must be fit on training data only, then applied to val and test — never fit on the full dataset.

---

