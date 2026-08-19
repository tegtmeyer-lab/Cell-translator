import torch
from cell_translator.training.hurdle_ae_training import masked_smooth_l1


def test_additive_eps_mode_matches_organoid_formula():
    pred = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[0.5, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])
    got = masked_smooth_l1(pred, target, mask, eps=1e-8, eps_mode="additive")
    base = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=0.1)
    weighted = base * (1.0 + torch.clamp(target, max=6.0))
    expected = (weighted * mask).sum() / mask.sum().clamp_min(1.0) + 1e-8
    assert torch.allclose(got, expected)
