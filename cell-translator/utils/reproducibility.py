"""Random-seed helpers used by the two framework-specific training stages."""

import numpy as np
import torch
try:
    import tensorflow as tf
except ImportError:  # permits PyTorch-only Stage 1 utilities to import independently
    tf = None

from cell_translator.hyperparameters import SEED

def set_torch_seed(seed: int = SEED):
    """Seed NumPy and PyTorch for HurdleAE training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_tf_seed(seed: int = SEED):
    """Seed NumPy and TensorFlow/Keras for CycleGAN training."""
    np.random.seed(seed)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(seed)
    else:  # compatibility with older TensorFlow/Keras releases
        tf.random.set_seed(seed)
