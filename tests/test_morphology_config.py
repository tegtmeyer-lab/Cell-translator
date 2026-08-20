from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "cell_painting_morphology"


def _load(name):
    common = yaml.safe_load((CFG / "common.yaml").read_text())
    specific = yaml.safe_load((CFG / name).read_text())
    common.update({k: v for k, v in specific.items() if k != "common_config" and not isinstance(v, dict)})
    return common, specific


def test_morphology_core_parameters():
    common = yaml.safe_load((CFG / "common.yaml").read_text())
    ae = common["morphology_ae"]
    gan = common["cyclegan"]
    assert ae["latent_dim"] == 256
    assert ae["epochs"] == 100
    assert ae["batch_size"] == 128
    assert ae["lambda_variance"] == 2.0
    assert ae["lambda_covariance"] == 0.1
    assert ae["torch_seed"] is None
    assert gan["batch_size"] == 32
    assert gan["epochs"] == 300
    assert gan["mmd_sigmas"] == [0.5, 1.0, 2.0]
    assert gan["tensorflow_seed"] is None
    assert common["features"]["expected_n_features"] == 3409


def test_deletion_neuron_path_is_not_exposed():
    deletion = yaml.safe_load((CFG / "deletion_line45.yaml").read_text())
    labels = [x["label"] for x in deletion["transitions"] if x.get("enabled", True)]
    assert "NPC → Neuron" not in labels
    assert labels == ["Stem → NPC", "NPC → Astrocyte"]


def test_transition_specific_validation_modes():
    control = yaml.safe_load((CFG / "control_line24.yaml").read_text())
    by_label = {x["label"]: x for x in control["transitions"]}
    assert by_label["Stem → NPC"]["evaluation"] == {
        "correlation_sampling": "full",
        "ks_sampling": "full",
        "std_ddof": 1,
        "nan_aware": True,
        "nn_method": "train_common_scaled_5nn",
    }
    assert by_label["NPC → Neuron"]["evaluation"]["ks_sampling"] == "equal"
    assert by_label["NPC → Neuron"]["evaluation"]["nn_method"] == "validation_unscaled_5nn"
    assert by_label["NPC → Astrocyte"]["evaluation"]["nn_method"] == "validation_unscaled_5nn"

    deletion = yaml.safe_load((CFG / "deletion_line45.yaml").read_text())
    for transition in deletion["transitions"]:
        assert transition["evaluation"]["correlation_sampling"] == "equal"
        assert transition["evaluation"]["ks_sampling"] == "equal"
        assert transition["evaluation"]["nn_method"] == "validation_unscaled_1nn"
