"""CycleGAN generator and critic architectures.

Extracted without scientific changes from the reviewer-ready reference script.
"""

from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, LayerNormalization

def build_generator(input_dim, output_dim, name):
    x_in = Input(shape=(input_dim,))
    x = Dense(1024, activation="relu")(x_in); x = LayerNormalization()(x)
    x = Dense(512, activation="relu")(x); x = LayerNormalization()(x)
    x = Dense(512, activation="relu")(x); x = LayerNormalization()(x)
    out = Dense(output_dim)(x)
    return models.Model(x_in, out, name=name)
def build_discriminator(input_dim, name):
    x_in = Input(shape=(input_dim,))
    x = Dense(512, activation="relu")(x_in)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    out = Dense(1)(x)   # WGAN critic logit (no sigmoid)
    return models.Model(x_in, out, name=name)
def build_generator_with_affine_head(input_dim, output_dim, name_prefix):
    """Forward (source -> target) generator: base MLP + a bare affine
    calibration layer on top, as in the original notebook. Only the
    forward generator in each direction gets this extra head; the
    backward (target -> source) generator does not."""
    core = build_generator(input_dim, output_dim, name=f"{name_prefix}_core")
    x_in = layers.Input(shape=(input_dim,))
    y = core(x_in)
    y = layers.Dense(output_dim, use_bias=True, name="linear_head")(y)
    return models.Model(x_in, y, name=name_prefix)
