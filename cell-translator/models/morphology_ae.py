"""Autoencoder architecture for continuous Cell Painting morphology profiles.

This module contains the dense morphology autoencoder used by the Cell Painting
application. It is separate from the HurdleAE models used for sparse scRNA-seq
expression because Cell Painting features are dense, continuous measurements.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MorphologyAutoencoder(nn.Module):
    """Batch-normalized autoencoder for standardized morphology features."""

    def __init__(self, input_dim: int, latent_dim: int = 256):
        super().__init__()

        def enc_block(d_in: int, d_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.BatchNorm1d(d_out),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.15),
            )

        def dec_block(d_in: int, d_out: int, drop: bool = True) -> nn.Sequential:
            block = [
                nn.Linear(d_in, d_out),
                nn.BatchNorm1d(d_out),
                nn.LeakyReLU(0.1),
            ]
            if drop:
                block.append(nn.Dropout(0.15))
            return nn.Sequential(*block)

        self.encoder = nn.Sequential(
            enc_block(input_dim, 1024),
            enc_block(1024, 512),
            nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(0.1),
            ),
            nn.Linear(256, latent_dim),
        )

        self.decoder = nn.Sequential(
            dec_block(latent_dim, 256),
            dec_block(256, 512),
            dec_block(512, 1024, drop=False),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
