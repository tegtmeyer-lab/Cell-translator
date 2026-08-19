import torch

from cell_translator.models.perturbation_hurdle_ae import (
    SeparateEncoderHurdleAE,
    remap_legacy_perturbation_state_dict,
)


def test_separate_encoder_hurdle_ae_shapes():
    model = SeparateEncoderHurdleAE(input_dim=31, latent_dim=8, dropout=0.10, target_domain="APOE")
    x = torch.rand(7, 31)
    for domain in ("control", "APOE", "target"):
        z, x_soft, logits, p, mag = model(x, domain)
        assert z.shape == (7, 8)
        assert x_soft.shape == logits.shape == p.shape == mag.shape == (7, 31)
        assert torch.all(mag >= 0)


def test_legacy_checkpoint_encoder_name_remap():
    legacy = {
        "encoder_ctrl.0.0.weight": torch.ones(2, 2),
        "encoder_apoe.0.0.weight": torch.zeros(2, 2),
        "decoder.mask_logits.weight": torch.ones(2, 2),
    }
    mapped = remap_legacy_perturbation_state_dict(legacy, "APOE")
    assert "encoder_ctrl.0.0.weight" in mapped
    assert "encoder_target.0.0.weight" in mapped
    assert "encoder_apoe.0.0.weight" not in mapped
    assert "decoder.mask_logits.weight" in mapped
