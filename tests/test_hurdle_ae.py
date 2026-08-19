import torch
from cell_translator.models.hurdle_ae import HurdleAutoencoder


def test_hurdle_ae_shapes():
    model = HurdleAutoencoder(input_dim=20, latent_dim=8, dropout=0.10)
    x = torch.rand(4, 20)
    z, x_soft, logits, p, mag = model(x)
    assert z.shape == (4, 8)
    assert x_soft.shape == logits.shape == p.shape == mag.shape == (4, 20)
