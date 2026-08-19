import pytest

pytest.importorskip("tensorflow")

from cell_translator.training.cyclegan_training import CycleGANTrainingConfig


def test_perturbation_cycle_config_overrides():
    cfg = CycleGANTrainingConfig.from_dict({
        "bank_sample": 512,
        "mmd_sigmas": [4.0, 8.0, 16.0, 32.0],
        "steps_per_epoch_from": "source",
        "target_shuffle_buffer": 512,
        "mnn_pool": 400,
        "mnn_target_replace": True,
        "tf_seed": None,
    })
    assert cfg.bank_sample == 512
    assert cfg.mmd_sigmas == (4.0, 8.0, 16.0, 32.0)
    assert cfg.steps_per_epoch_from == "source"
    assert cfg.tf_seed is None
