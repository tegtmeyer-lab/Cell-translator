from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "acute_leukemia"


def _load(name):
    with open(CFG / "common.yaml", encoding="utf-8") as f:
        common = yaml.safe_load(f)
    with open(CFG / name, encoding="utf-8") as f:
        specific = yaml.safe_load(f)
    return common, specific


def test_historical_training_parameters():
    common, _ = _load("healthy_cd34_to_bmmc.yaml")
    h = common["hurdle_ae"]
    assert h["latent_dim"] == 128
    assert h["batch_size"] == 256
    assert h["learning_rate"] == 0.001
    assert h["epochs"] == 200
    assert h["patience"] == 10
    assert h["input_drop_prob"] == 0.15
    c = common["cyclegan"]
    assert c["batch_size"] == 128
    assert c["lambda_cycle"] == 10.0
    assert c["lambda_identity"] == 0.1
    assert c["k_pair"] == 9
    assert c["bank_sample"] == 5000
    assert c["n_critic"] == 3
    assert c["pair_weight"] == 5.0
    assert c["mmd_weight"] == 2.0
    assert c["covariance_weight"] == 2.0
    assert c["tf_seed"] is None


def test_workflow_shapes_and_outputs():
    _, healthy = _load("healthy_cd34_to_bmmc.yaml")
    _, disease = _load("disease_cd34_to_mpal.yaml")
    assert healthy["expected_input"] == {"source_cells": 5752, "target_cells": 6270, "genes": 5000}
    assert disease["expected_input"] == {"source_cells": 5752, "target_cells": 5885, "genes": 5000}
    assert healthy["evaluation"]["distribution_output"] == "hard"
    assert disease["evaluation"]["distribution_output"] == "library_matched"
