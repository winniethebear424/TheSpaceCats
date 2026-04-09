
# Top-level CNN-AE model. Wraps encoder + decoder.
# forward() returns (reconstruction, latent) so we can
# extract latent vectors for t-SNE / KNN eval separately.



import torch.nn as nn
from models.encoder import CNNEncoder
from models.decoder import CNNDecoder
from config import CFG

class MaskedAutoencoder(nn.Module):
    def __init__(self,
                 in_channels=5,
                 hidden_dims=None,
                 latent_dim=None):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, hidden_dims, latent_dim)
        self.decoder = CNNDecoder(in_channels, hidden_dims, latent_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) — masked input image
        Returns:
            recon:  (B, C, H, W) — full reconstruction
            latent: (B, latent_dim) — embedding vector
        """
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        return recon, latent

    def encode(self, x):
        """Convenience method for evaluation — encoder only."""
        return self.encoder(x)