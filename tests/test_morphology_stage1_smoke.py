from pathlib import Path
import numpy as np

from cell_translator.training.morphology_ae_training import (
    MorphologyAETrainingConfig,
    run_morphology_ae_state,
    load_state_artifacts,
)


def test_morphology_stage1_one_epoch_smoke(tmp_path: Path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 16)).astype(np.float32)
    # Exercise the historical median-imputation path too.
    X[0, 0] = np.nan
    cfg = MorphologyAETrainingConfig(
        latent_dim=4,
        epochs=1,
        batch_size=8,
        patience=2,
        inference_batch_size=16,
        torch_seed=42,
    )
    metrics = run_morphology_ae_state(
        X,
        [f"feature_{i}" for i in range(X.shape[1])],
        "synthetic",
        str(tmp_path),
        cfg,
    )
    assert metrics["n_cells"] == 40
    assert metrics["best_epoch"] == 1
    artifacts = load_state_artifacts("synthetic", str(tmp_path))
    assert artifacts["Z_train"].shape[1] == 4
    assert artifacts["X_validation_scaled"].shape[1] == 16
