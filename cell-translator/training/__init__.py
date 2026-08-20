"""Training routines for Cell Translator.

PyTorch autoencoder training utilities are imported eagerly. TensorFlow-based
CycleGAN training utilities are exposed lazily so Stage 1 functionality remains
importable in environments without TensorFlow installed.
"""

from .hurdle_ae_training import (
    HurdleAETrainingConfig,
    run_shared_hurdle_ae,
    train_shared_hurdle_ae,
)
from .independent_hurdle_ae_training import (
    IndependentHurdleAETrainingConfig,
    run_independent_hurdle_ae,
    train_independent_hurdle_ae,
)
from .morphology_ae_training import (
    MorphologyAETrainingConfig,
    run_morphology_ae_state,
    train_morphology_ae,
)
from .perturbation_hurdle_ae_training import (
    run_perturbation_hurdle_ae,
    train_separate_encoder_ae,
)

__all__ = [
    "HurdleAETrainingConfig",
    "run_shared_hurdle_ae",
    "train_shared_hurdle_ae",
    "IndependentHurdleAETrainingConfig",
    "run_independent_hurdle_ae",
    "train_independent_hurdle_ae",
    "MorphologyAETrainingConfig",
    "run_morphology_ae_state",
    "train_morphology_ae",
    "run_perturbation_hurdle_ae",
    "train_separate_encoder_ae",
    "CycleGANTrainingConfig",
    "train_latent_cyclegan",
    "make_cyclegan_train_step",
    "plot_gan_training_curves",
    "AcuteLeukemiaCycleGANConfig",
    "train_acute_leukemia_cyclegan",
    "MorphologyCycleGANConfig",
    "train_morphology_cyclegan",
]


def __getattr__(name):
    common = {
        "CycleGANTrainingConfig",
        "train_latent_cyclegan",
        "make_cyclegan_train_step",
        "plot_gan_training_curves",
    }
    leukemia = {"AcuteLeukemiaCycleGANConfig", "train_acute_leukemia_cyclegan"}
    morphology = {"MorphologyCycleGANConfig", "train_morphology_cyclegan"}

    if name in leukemia:
        from . import acute_leukemia_cyclegan_training as module
        return getattr(module, name)
    if name in morphology:
        from . import morphology_cyclegan_training as module
        return getattr(module, name)
    if name in common:
        from . import cyclegan_training as module
        return getattr(module, name)
    raise AttributeError(name)
