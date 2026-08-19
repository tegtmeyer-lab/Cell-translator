"""Training routines for Cell Translator.

PyTorch HurdleAE training utilities are imported eagerly. TensorFlow-based
CycleGAN training utilities are exposed lazily so Stage 1 functionality
remains importable in environments without TensorFlow installed.
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
