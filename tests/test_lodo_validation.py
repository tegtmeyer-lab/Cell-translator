import numpy as np
import yaml
import torch
from analyses.leave_one_line_out.models import LodoMorphologyAutoencoder
from analyses.leave_one_line_out.training import lodo_morphology_loss
from analyses.leave_one_line_out.evaluation import evaluate_fold_arrays, pca_variance_error


def test_lodo_ae_shape_and_loss():
    model=LodoMorphologyAutoencoder(input_dim=20,latent_dim=8,dropout=0.03)
    x=torch.randn(16,20)
    z,recon=model(x)
    assert z.shape==(16,8)
    assert recon.shape==x.shape
    assert torch.isfinite(lodo_morphology_loss(recon,x))


def test_lodo_metrics_smoke():
    rng=np.random.default_rng(42)
    source=rng.normal(size=(100,15)).astype(np.float32)
    measured=(source[:80]+0.3+rng.normal(scale=.1,size=(80,15))).astype(np.float32)
    predicted=(source[:90]+0.25+rng.normal(scale=.12,size=(90,15))).astype(np.float32)
    out=evaluate_fold_arrays(source,measured,predicted,ks_max_cells=60,pca_components=5,pca_log_base=2)
    assert np.isfinite(out["median_ks"])
    assert np.isfinite(out["median_abs_log_pca_variance_ratio"])
    assert 0 <= out["auc_distance_from_0p5"] <= 0.5


def test_lodo_selected_lines_config():
    with open("configs/validation/leave_one_line_out.yaml","r",encoding="utf-8") as h:
        cfg=yaml.safe_load(h)
    assert cfg["excluded_line_ids"] == [22,23,24,25]
    assert cfg["selected_heldout_lines"]["stem_to_npc"] == [10,15,20,21,9]
    assert cfg["autoencoder"]["latent_dim"] == 128
    assert cfg["cyclegan"]["n_critic"] == 3
