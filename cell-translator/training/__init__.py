"""Training routines for Cell Translator.

The PyTorch HurdleAE training API is imported eagerly. TensorFlow CycleGAN
training is exposed lazily so Stage 1 utilities remain importable in a
PyTorch-only environment.
"""

from .hurdle_ae_training import run_shared_hurdle_ae, train_shared_hurdle_ae

__all__ = [
    "run_shared_hurdle_ae",
    "train_shared_hurdle_ae",
    "train_latent_cyclegan",
    "make_cyclegan_train_step",
    "plot_gan_training_curves",
]


def __getattr__(name):
    if name in {
        "train_latent_cyclegan",
        "make_cyclegan_train_step",
        "plot_gan_training_curves",
    }:
        from . import cyclegan_training
        return getattr(cyclegan_training, name)
    raise AttributeError(name)
