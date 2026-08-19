"""Hurdle Autoencoder architecture.

Extracted without scientific changes from the reviewer-ready reference script.
"""

import torch
import torch.nn as nn

class HurdleAutoencoder(nn.Module):
    """
    Two-head decoder:
      logits  -> p_detect  (sigmoid) -- WHAT is expressed?
      mag_raw -> mag (softplus)      -- HOW MUCH is expressed?
    x_soft = p * mag  (soft reconstruction)
    x_hard = 1(p > tau_gene) * mag (hard reconstruction, used at inference)
    """

    def __init__(self, input_dim: int, latent_dim: int = 64, dropout: float = 0.10):
        super().__init__()

        def enc_block(d_in, d_out):
            return nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.LayerNorm(d_out),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def dec_block(d_in, d_out):
            return nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.LayerNorm(d_out),
                nn.GELU(),
            )

        self.encoder_net = nn.Sequential(
            enc_block(input_dim, 512),
            enc_block(512, 256),
            nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
            ),
            nn.Linear(128, latent_dim),
        )

        self.dec_trunk = nn.Sequential(
            dec_block(latent_dim, 128),
            dec_block(128, 256),
            dec_block(256, 512),
        )

        self.mask_logits = nn.Linear(512, input_dim)
        self.mag_raw = nn.Linear(512, input_dim)
        self.softplus = nn.Softplus()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_net(x)

    def decode(self, z: torch.Tensor):
        h = self.dec_trunk(z)
        logits = self.mask_logits(h)             # (B, G)
        p = torch.sigmoid(logits)                # (B, G) detection probability
        mag = self.softplus(self.mag_raw(h))     # (B, G) always >= 0
        x_soft = p * mag
        return x_soft, logits, p, mag

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_soft, logits, p, mag = self.decode(z)
        return z, x_soft, logits, p, mag
