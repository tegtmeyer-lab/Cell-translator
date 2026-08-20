"""Latent-space CycleGAN architectures for Cell Painting morphology translation.

The morphology application uses a 256-dimensional generator/critic architecture
that differs from the transcriptomic Cell Translator CycleGAN architecture.
"""
from __future__ import annotations


def _tf():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.layers import Input, Dense, LayerNormalization
    return tf, layers, models, Input, Dense, LayerNormalization


def build_morphology_generator(input_dim: int, output_dim: int, name: str):
    _, _, models, Input, Dense, LayerNormalization = _tf()
    x = inp = Input(shape=(input_dim,))
    x = LayerNormalization()(Dense(512, activation="relu")(x))
    x = LayerNormalization()(Dense(256, activation="relu")(x))
    x = LayerNormalization()(Dense(256, activation="relu")(x))
    x = Dense(output_dim)(x)
    return models.Model(inp, x, name=name)


def build_morphology_generator_with_head(input_dim: int, output_dim: int, name: str):
    _, layers, models, _, _, _ = _tf()
    core = build_morphology_generator(input_dim, output_dim, name=f"{name}_core")
    inp = layers.Input(shape=(input_dim,))
    y = core(inp)
    y = layers.Dense(output_dim, use_bias=True, name="linear_head")(y)
    return models.Model(inp, y, name=name)


def build_morphology_discriminator(input_dim: int, name: str):
    _, _, models, Input, Dense, _ = _tf()
    x = inp = Input(shape=(input_dim,))
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1)(x)
    return models.Model(inp, x, name=name)
