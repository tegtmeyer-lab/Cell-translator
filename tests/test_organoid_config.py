from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "cerebral_organoid"


def load(name):
    with open(CFG / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_organoid_common_parameters_match_source_notebook():
    cfg = load("common.yaml")
    assert cfg["model"]["latent_dim"] == 128
    ae = cfg["hurdle_ae"]
    assert ae["batch_size"] == 128
    assert ae["lr"] == 5e-4
    assert ae["epochs"] == 300
    assert ae["masked_smooth_l1_eps_mode"] == "additive"
    gan = cfg["cyclegan"]
    assert gan["batch_size"] == 32
    assert gan["epochs"] == 300
    assert gan["bank_sample"] == 8192
    assert gan["mmd_sigmas"] == [1.0, 2.0, 4.0]
    assert gan["steps_per_epoch_from"] == "target"
    assert gan["moment_eps_mode"] == "difference"
    assert gan["tf_seed"] is None
    assert cfg["translation"]["split"] == "validation"


def test_organoid_dataset_pairs_and_hvgs():
    p = load("primary_scp1756.yaml")
    h = load("hnoca_he.yaml")
    t = load("third_organoid.yaml")
    assert (p["source_state"], p["target_state"], p["preparation"]["n_hvg"]) == ("2month", "6month", 15000)
    assert (h["source_state"], h["target_state"], h["preparation"]["n_hvg"]) == ("15d", "60d", 5000)
    assert h["preparation"]["selected_ages_days"] == [15,30,60,90,120,150,180]
    assert (t["source_state"], t["target_state"], t["preparation"]["n_hvg"]) == ("18d", "61d", 5000)
    assert t["preparation"]["selected_ages_days"] == [4,9,12,18,21,61]
