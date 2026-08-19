import numpy as np
import torch
from torch.utils.data import DataLoader

from cell_translator.models.perturbation_hurdle_ae import SeparateEncoderHurdleAE
from cell_translator.training.hurdle_ae_training import NumpyExpressionDataset, compute_pos_weight_per_gene
from cell_translator.training.perturbation_hurdle_ae_training import train_separate_encoder_ae


def test_perturbation_stage1_one_epoch_smoke():
    rng = np.random.default_rng(42)
    ctrl = rng.poisson(0.4, size=(16, 12)).astype(np.float32)
    target = rng.poisson(0.5, size=(16, 12)).astype(np.float32)
    train_ctrl, val_ctrl = ctrl[:12], ctrl[12:]
    train_target, val_target = target[:12], target[12:]

    def loader(x, shuffle, drop_last):
        return DataLoader(NumpyExpressionDataset(x), batch_size=4, shuffle=shuffle, drop_last=drop_last)

    model = SeparateEncoderHurdleAE(input_dim=12, latent_dim=8, dropout=0.10, target_domain="APOE")
    history = train_separate_encoder_ae(
        model,
        loader(train_ctrl, True, True),
        loader(train_target, True, True),
        loader(val_ctrl, False, False),
        loader(val_target, False, False),
        torch.device("cpu"),
        compute_pos_weight_per_gene(train_ctrl),
        compute_pos_weight_per_gene(train_target),
        config={"epochs": 1, "patience": 1},
    )
    assert history["best_epoch"] == 1
    assert np.isfinite(history["best_val"])
