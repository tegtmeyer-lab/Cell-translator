from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "configs" / "genetic_perturbation" / "common.yaml"


def test_perturbation_training_parameters_are_explicit():
    cfg = yaml.safe_load(COMMON.read_text())
    ae = cfg["hurdle_ae"]
    assert ae["seed"] == 42
    assert ae["latent_dim"] == 128
    assert ae["batch_size"] == 64
    assert ae["learning_rate"] == 0.0005
    assert ae["epochs"] == 300
    assert ae["patience"] == 25
    assert ae["validation_fraction"] == 0.20
    assert ae["train_drop_last"] is True

    gan = cfg["cyclegan"]
    assert gan["batch_size"] == 32
    assert gan["epochs"] == 300
    assert gan["bank_sample"] == 512
    assert gan["mmd_sigmas"] == [4.0, 8.0, 16.0, 32.0]
    assert gan["steps_per_epoch_from"] == "source"
    assert gan["source_shuffle_buffer"] == 4096
    assert gan["target_shuffle_buffer"] == 512
    assert gan["tf_seed"] is None
