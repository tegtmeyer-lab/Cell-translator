from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "pancreatic_endocrine"


def load(name):
    with open(CFG / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_pancreatic_parameters_match_source_notebook():
    common = load("common.yaml")
    assert common["model"]["latent_dim"] == 64
    ae = common["hurdle_ae"]
    assert ae["batch_size"] == 64
    assert ae["lr"] == 5e-4
    assert ae["epochs"] == 300
    assert ae["masked_smooth_l1_eps_mode"] == "additive"

    gan = common["cyclegan"]
    assert gan["batch_size"] == 32
    assert gan["bank_sample"] == 8192
    assert gan["mmd_sigmas"] == [1.0, 2.0, 4.0]
    assert gan["moment_eps_mode"] == "symmetric"
    assert gan["steps_per_epoch_from"] == "target"
    assert gan["tf_seed"] is None
    assert common["translation"]["split"] == "validation"


def test_pancreatic_final_preparation_is_5003_gene_path():
    cfg = load("gse132188_ngn3_to_endocrine.yaml")
    prep = cfg["preparation"]
    assert cfg["source_state"] == "NGN3"
    assert cfg["target_state"] == "Endocrine"
    assert prep["source_cluster_label"] == "Ngn3 high EP"
    assert prep["source_day_max"] == 14.5
    assert prep["target_day_min"] == 14.5
    assert prep["hvg_flavor"] == "seurat"
    assert prep["n_hvg_base"] == 5000
    assert prep["expected_force_added_markers"] == ["Gcg", "Chgb", "Kcnj11"]
    assert prep["expected_final_gene_count"] == 5003
    assert prep["expected_cell_counts"] == {"source": 1861, "target": 2621}
