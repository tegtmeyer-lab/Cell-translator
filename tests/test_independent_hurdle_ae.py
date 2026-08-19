import torch
from cell_translator.models.independent_hurdle_ae import IndependentHurdleAutoencoder


def test_independent_hurdle_shapes():
    model = IndependentHurdleAutoencoder(input_dim=31, latent_dim=128, dropout=0.10)
    x = torch.rand(7, 31)
    z, x_hat, logits, p, mag = model(x)
    assert z.shape == (7, 128)
    for a in (x_hat, logits, p, mag):
        assert a.shape == (7, 31)
    assert torch.all(mag >= 0)
