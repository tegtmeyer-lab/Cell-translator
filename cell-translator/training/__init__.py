"""Training routines for Cell Translator.

PyTorch HurdleAE training utilities are imported eagerly. TensorFlow-based
CycleGAN training utilities are exposed lazily so Stage 1 functionality
remains importable in environments without TensorFlow installed.
"""

from .hurdle_ae_training import HurdleAETrainingConfig, run_shared_hurdle_ae, train_shared_hurdle_ae

from .independent_hurdle_ae_training import (
    IndependentHurdleAETrainingConfig,
    run_independent_hurdle_ae,
    train_independent_hurdle_ae,
)
from .perturbation_hurdle_ae_training import (
    run_perturbation_hurdle_ae,
    train_separate_encoder_ae,
)

__all__ = [
    "IndependentHurdleAETrainingConfig",
    "run_independent_hurdle_ae",
    "train_independent_hurdle_ae",
    "AcuteLeukemiaCycleGANConfig",
    "train_acute_leukemia_cyclegan",
    "HurdleAETrainingConfig",
    "run_shared_hurdle_ae",
    "train_shared_hurdle_ae",
    "run_perturbation_hurdle_ae",
    "train_separate_encoder_ae",
    "CycleGANTrainingConfig",
    "train_latent_cyclegan",
    "make_cyclegan_train_step",
    "plot_gan_training_curves",
]


def __getattr__(name):
    if name in {
        "CycleGANTrainingConfig",
        "train_latent_cyclegan",
        "make_cyclegan_train_step",
        "plot_gan_training_curves",
        "AcuteLeukemiaCycleGANConfig",
        "train_acute_leukemia_cyclegan",
    }:
        if name in {"AcuteLeukemiaCycleGANConfig", "train_acute_leukemia_cyclegan"}:
            from . import acute_leukemia_cyclegan_training as module
        else:
            from . import cyclegan_training as module
        return getattr(module, name)
    raise AttributeError(name)
