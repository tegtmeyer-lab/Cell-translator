"""Validation-specific reduced AE and CycleGAN architectures used for LOLO.

These models intentionally remain analysis-local because they are not the
canonical 256-dimensional morphology models used in the main application.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class LodoMorphologyAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128, dropout: float = 0.03):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.1),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.1),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def build_lodo_generator(input_dim: int, output_dim: int, name: str):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    x_in = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation="relu")(x_in); x = layers.LayerNormalization()(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.LayerNormalization()(x)
    x = layers.Dense(256, activation="relu")(x); x = layers.LayerNormalization()(x)
    out = layers.Dense(output_dim)(x)
    return models.Model(x_in, out, name=name)


def build_lodo_discriminator(input_dim: int, name: str):
    from tensorflow.keras import layers, models
    x_in = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation="relu")(x_in)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1)(x)
    return models.Model(x_in, out, name=name)
