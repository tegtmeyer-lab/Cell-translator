"""Model architectures for Cell Translator."""

from .hurdle_ae import HurdleAutoencoder
from .independent_hurdle_ae import IndependentHurdleAutoencoder
from .perturbation_hurdle_ae import (
    SeparateEncoderHurdleAE,
    SharedHurdleDecoder,
    remap_legacy_perturbation_state_dict,
)

__all__ = [
    "HurdleAutoencoder",
    "IndependentHurdleAutoencoder",
    "SeparateEncoderHurdleAE",
    "SharedHurdleDecoder",
    "remap_legacy_perturbation_state_dict",
]
