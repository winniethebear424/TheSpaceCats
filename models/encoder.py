# CNN Encoder: (B, C, 64, 64) -> (B, latent_dim)
# Stacked conv blocks compress spatial info into a
# compact latent vector that captures galaxy morphology.

import torch.nn as nn
from config import CFG

def conv_block(in_ch, out_ch, k=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=2, padding=k//2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class CNNEncoder(nn.Module):
    def __init__(self,
                 in_channels=5,
                 hidden_dims=None,
                 latent_dim=None):
        super().__init__()
        hidden_dims = hidden_dims or CFG.hidden_dims
        latent_dim  = latent_dim  or CFG.latent_dim

        layers, ch = [], in_channels
        for h in hidden_dims:
            layers.append(conv_block(ch, h, CFG.kernel_size))
            ch = h
        self.conv_blocks = nn.Sequential(*layers)

        # spatial size after N stride-2 convs on 64x64 input
        spatial = CFG.image_size // (2 ** len(hidden_dims))
        self.flatten = nn.Flatten() # (B, 128, 8, 8)  →  (B, 128*8*8)  =  (B, 8192)
        self.project = nn.Linear(ch * spatial * spatial, latent_dim) # (B, 8192)  →  (B, 128)

    def forward(self, x):
        x = self.conv_blocks(x)   # (B, hidden_dims[-1], s, s)
        x = self.flatten(x)        # (B, hidden_dims[-1]*s*s)
        return self.project(x)     # (B, latent_dim)