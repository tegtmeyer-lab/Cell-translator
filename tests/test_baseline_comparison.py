import numpy as np
from analyses.baseline_comparison.run import (
    identity_baseline, mean_shift_baseline, nearest_neighbor_baseline,
    linear_latent_shift_baseline, evaluate_prediction,
)


def test_baselines_and_metrics_smoke():
    rng=np.random.default_rng(42)
    src_tr=rng.normal(size=(80,12)).astype(np.float32)
    tgt_tr=(src_tr[:60]+0.2).astype(np.float32)
    src_ev=rng.normal(size=(40,12)).astype(np.float32)
    tgt_ev=(rng.normal(size=(45,12))+0.2).astype(np.float32)
    preds=[identity_baseline(src_ev),mean_shift_baseline(src_tr,tgt_tr,src_ev),
           nearest_neighbor_baseline(src_tr,tgt_tr,src_ev,5,42),
           linear_latent_shift_baseline(src_tr,tgt_tr,src_ev,5,42)]
    for pred in preds:
        assert pred.shape==src_ev.shape
        out=evaluate_prediction("x","m",src_ev,tgt_ev,pred,top_k=5,ks_max_genes=8,auc_cells=30)
        assert "median_gene_ks" in out
