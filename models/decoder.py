

# CNN Decoder: (B, latent_dim) -> (B, C, 64, 64)
# Mirrors the encoder with transposed convolutions.
# Only the masked patch pixels are used in the loss.

import torch.nn as nn
from config import CFG

def deconv_block(in_ch, out_ch, k=3):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, k,
                           stride=2, padding=k//2, output_padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class CNNDecoder(nn.Module):
    def __init__(self,
                 out_channels=5,
                 hidden_dims=None,
                 latent_dim=None):
        super().__init__()
        hidden_dims = list(reversed(hidden_dims or CFG.hidden_dims))
        latent_dim  = latent_dim or CFG.latent_dim

        spatial = CFG.image_size // (2 ** len(hidden_dims))
        self.spatial = spatial
        self.first_ch = hidden_dims[0]
        self.unproject = nn.Linear(latent_dim, hidden_dims[0] * spatial * spatial)

        layers, ch = [], hidden_dims[0]
        for h in hidden_dims[1:]:
            layers.append(deconv_block(ch, h, CFG.kernel_size))
            ch = h
        layers.append(nn.ConvTranspose2d(
            ch, out_channels, CFG.kernel_size,
            stride=2, padding=CFG.kernel_size//2, output_padding=1
        ))
        layers.append(nn.Sigmoid())  # output in [0, 1]
        self.deconv_blocks = nn.Sequential(*layers)

    def forward(self, z):
        x = self.unproject(z)
        x = x.view(-1, self.first_ch, self.spatial, self.spatial)
        return self.deconv_blocks(x)   # (B, C, 64, 64)