"""Independent-state HurdleAE architecture used in acute leukemia analyses.

Each biological state is modeled with its own HurdleAE. The architecture
preserves the encoder/decoder structure used for the reported CD34+ -> BMMC
and CD34+ -> MPAL analyses.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IndependentHurdleAutoencoder(nn.Module):
    """Two-head HurdleAE trained independently for one cell state."""

    def __init__(self, input_dim: int, latent_dim: int = 128, dropout: float = 0.10):
        super().__init__()

        def enc_block(d_in: int, d_out: int):
            return nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.LayerNorm(d_out),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        def dec_block(d_in: int, d_out: int):
            return nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.LayerNorm(d_out),
                nn.GELU(),
            )

        self.encoder = nn.Sequential(
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
        )
        self.mask_logits = nn.Linear(256, input_dim)
        self.mag_raw = nn.Linear(256, input_dim)
        self.softplus = nn.Softplus()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor):
        h = self.dec_trunk(z)
        logits = self.mask_logits(h)
        p = torch.sigmoid(logits)
        mag = self.softplus(self.mag_raw(h))
        x_hat = p * mag
        return x_hat, logits, p, mag

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat, logits, p, mag = self.decode(z)
        return z, x_hat, logits, p, mag
