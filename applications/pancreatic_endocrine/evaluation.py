"""Evaluation metrics for NGN3-high-EP to Endocrine translation."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, pearsonr, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity


def _state_slices(npz) -> dict[str, tuple[int, int]]:
    if not all(k in npz.files for k in ("state_names", "slice_starts", "slice_ends")):
        raise KeyError("AE outputs must contain saved state slice metadata.")
    names = [str(x) for x in npz["state_names"].tolist()]
    starts = npz["slice_starts"].astype(int)
    ends = npz["slice_ends"].astype(int)
    return {name: (int(s), int(e)) for name, s, e in zip(names, starts, ends)}


def evaluate_translation(
    target_matrix: str,
    ae_npz: str,
    translated_npz: str,
    target_state: str,
    threshold: float = 1e-6,
    seed: int = 42,
) -> dict:
    ae = np.load(ae_npz, allow_pickle=False)
    translated = np.load(translated_npz, allow_pickle=False)
    slices = _state_slices(ae)
    if target_state not in slices:
        raise KeyError(f"Target state {target_state!r} is absent from AE metadata.")

    val_idx = ae["val_idx"] if "val_idx" in ae.files else ae["test_idx"]
    s, e = slices[target_state]
    target_global = np.arange(s, e, dtype=np.int64)
    target_val_global = np.intersect1d(target_global, val_idx)
    target_val_local = target_val_global - s

    x_target_all = np.load(target_matrix).astype(np.float32)
    x_real = x_target_all[target_val_local]
    x_fake = translated["X_translated"].astype(np.float32)
    z_real = ae["Z_all"][target_val_global].astype(np.float32)
    z_fake = translated["translated_z"].astype(np.float32)

    if x_real.shape[1] != x_fake.shape[1]:
        raise ValueError(f"Gene dimension mismatch: real={x_real.shape}, translated={x_fake.shape}.")
    if z_real.shape[1] != z_fake.shape[1]:
        raise ValueError(f"Latent dimension mismatch: real={z_real.shape}, translated={z_fake.shape}.")

    # Stable latent summaries from the final evaluation notebook.
    mean_real, mean_fake = z_real.mean(0), z_fake.mean(0)
    std_real, std_fake = z_real.std(0, ddof=0), z_fake.std(0, ddof=0)
    latent_w1 = np.array(
        [wasserstein_distance(z_real[:, j], z_fake[:, j]) for j in range(z_real.shape[1])],
        dtype=float,
    )
    metrics = {
        "n_real_target_validation_cells": int(x_real.shape[0]),
        "n_translated_source_validation_cells": int(x_fake.shape[0]),
        "latent_mean_pearson_r": float(pearsonr(mean_real, mean_fake).statistic),
        "latent_std_pearson_r": float(pearsonr(std_real, std_fake).statistic),
        "latent_mean_wasserstein": float(latent_w1.mean()),
        "latent_mean_cosine_similarity": float(
            cosine_similarity(mean_real[None, :], mean_fake[None, :])[0, 0]
        ),
        "latent_mean_l2": float(np.linalg.norm(mean_real - mean_fake)),
        "latent_mean_mae": float(np.mean(np.abs(mean_real - mean_fake))),
        "latent_std_l2": float(np.linalg.norm(std_real - std_fake)),
        "latent_std_mae": float(np.mean(np.abs(std_real - std_fake))),
        "latent_std_relative_mae": float(
            np.mean(np.abs(std_real - std_fake) / (np.abs(std_real) + 1e-8))
        ),
        "real_target_zero_fraction": float((x_real <= threshold).mean()),
        "translated_zero_fraction": float((x_fake <= threshold).mean()),
        "gene_mean_pearson_r": float(pearsonr(x_real.mean(0), x_fake.mean(0)).statistic),
        "detection_rate_pearson_r": float(
            pearsonr((x_real > threshold).mean(0), (x_fake > threshold).mean(0)).statistic
        ),
    }

    # The reported gene-space KS/W1 analysis equalized cell counts using a
    # NumPy Generator seeded with 42.
    rng = np.random.default_rng(seed)
    n = min(x_real.shape[0], x_fake.shape[0])
    real = x_real[rng.choice(x_real.shape[0], n, replace=False)]
    fake = x_fake[rng.choice(x_fake.shape[0], n, replace=False)]
    n_genes = real.shape[1]
    ks_stats = np.empty(n_genes, dtype=np.float32)
    ks_pvals = np.empty(n_genes, dtype=np.float64)
    w1_stats = np.empty(n_genes, dtype=np.float32)
    mean_diffs = np.empty(n_genes, dtype=np.float32)

    for j in range(n_genes):
        ks = ks_2samp(real[:, j], fake[:, j])
        ks_stats[j] = ks.statistic
        ks_pvals[j] = ks.pvalue
        w1_stats[j] = wasserstein_distance(real[:, j], fake[:, j])
        mean_diffs[j] = float(fake[:, j].mean() - real[:, j].mean())

    metrics.update({
        "median_gene_ks": float(np.median(ks_stats)),
        "median_gene_wasserstein": float(np.median(w1_stats)),
        "median_abs_gene_mean_difference": float(np.median(np.abs(mean_diffs))),
        "fraction_gene_ks_lt_0_05": float(np.mean(ks_stats < 0.05)),
        "fraction_gene_ks_gt_0_20": float(np.mean(ks_stats > 0.20)),
        "fraction_gene_ks_gt_0_40": float(np.mean(ks_stats > 0.40)),
        "high_ks_gene_count_gt_0_30": int(np.sum(ks_stats > 0.30)),
    })
    return metrics


def save_metrics(metrics: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def print_metrics(metrics: dict) -> None:
    print("\n=== NGN3 -> Endocrine validation ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
