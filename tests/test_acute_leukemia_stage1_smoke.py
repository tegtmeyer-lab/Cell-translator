import numpy as np
import torch
from torch.utils.data import DataLoader

from cell_translator.models.independent_hurdle_ae import IndependentHurdleAutoencoder
from cell_translator.training.independent_hurdle_ae_training import (
    IndependentHurdleAETrainingConfig,
    NumpyExpressionDataset,
    train_independent_hurdle_ae,
)


def test_stage1_one_epoch_smoke():
    rng = np.random.default_rng(7)
    x = rng.exponential(0.5, size=(32, 20)).astype(np.float32)
    x[rng.random(x.shape) < 0.7] = 0.0
    train = DataLoader(NumpyExpressionDataset(x[:24]), batch_size=8, shuffle=True)
    val = DataLoader(NumpyExpressionDataset(x[24:]), batch_size=8, shuffle=False)
    model = IndependentHurdleAutoencoder(20, latent_dim=8, dropout=0.10)
    cfg = IndependentHurdleAETrainingConfig(latent_dim=8, batch_size=8, epochs=1, patience=1)
    hist = train_independent_hurdle_ae(model, train, val, torch.device("cpu"), cfg)
    assert hist["best_epoch"] == 1
    assert np.isfinite(hist["best_val"])
