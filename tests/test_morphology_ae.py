import torch

from cell_translator.models.morphology_ae import MorphologyAutoencoder
from cell_translator.training.morphology_ae_training import morphology_loss


def test_morphology_ae_shapes():
    model = MorphologyAutoencoder(input_dim=24, latent_dim=8)
    model.eval()
    x = torch.randn(4, 24)
    with torch.no_grad():
        z, recon = model(x)
    assert z.shape == (4, 8)
    assert recon.shape == x.shape


def test_morphology_loss_is_finite():
    real = torch.randn(8, 12)
    recon = real + 0.1 * torch.randn(8, 12)
    value = morphology_loss(recon, real, lambda_variance=2.0, lambda_covariance=0.1)
    assert torch.isfinite(value)
