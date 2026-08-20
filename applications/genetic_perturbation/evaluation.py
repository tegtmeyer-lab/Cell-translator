"""Evaluation metrics for genetic perturbation translation.

(1) per-gene distribution fidelity on held-out cells, (2) the historical LFC
summary used in the single-perturbation result suite, and (3) the mean-log1p
shift used in the final multi-model comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr, spearmanr, wasserstein_distance
from statsmodels.stats.multitest import multipletests


def per_gene_ks(X_real, X_pred, gene_names):
    stats = np.empty(X_real.shape[1], dtype=np.float64)
    pvals = np.empty_like(stats)
    for g in range(X_real.shape[1]):
        stats[g], pvals[g] = ks_2samp(X_real[:, g], X_pred[:, g])
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    return pd.DataFrame({"gene": gene_names, "ks_stat": stats, "pval": pvals, "pval_adj": qvals})


def per_gene_wasserstein(X_real, X_pred, gene_names):
    vals = np.asarray(
        [wasserstein_distance(X_real[:, g], X_pred[:, g]) for g in range(X_real.shape[1])],
        dtype=np.float64,
    )
    return pd.DataFrame({"gene": gene_names, "wasserstein": vals})


def historical_lfc_log1p_of_mean(X_a, X_b):
    """LFC formula used in the notebook's single-perturbation result suites.

    The input arrays are already log1p values; the original code computes
    ``log1p(mean(B)) - log1p(mean(A))``. The name is intentionally explicit so
    this historical metric is not confused with the dedicated linear-space
    log2FC analysis.
    """
    return np.log1p(X_b.mean(axis=0)) - np.log1p(X_a.mean(axis=0))


def linear_space_log2fc(X_source_log1p, X_target_log1p, pseudocount=1e-3):
    """Dedicated notebook log2FC: inverse log1p, average, then log2 ratio."""
    source_mean = np.expm1(X_source_log1p).mean(axis=0)
    target_mean = np.expm1(X_target_log1p).mean(axis=0)
    return np.log2((target_mean + pseudocount) / (source_mean + pseudocount))


def mean_log1p_shift(X_target, X_source):
    """Effect metric used in the final cross-model comparison."""
    return X_target.mean(axis=0) - X_source.mean(axis=0)


def evaluate_heldout_translation(
    X_source_val,
    X_target_val,
    X_pred,
    gene_names,
    threshold=1e-6,
    output_dir=None,
):
    """Reproduce the core held-out metrics printed by the per-gene notebook sections."""
    gene_names = np.asarray(gene_names).astype(str)
    if X_target_val.shape[1] != X_pred.shape[1] or X_source_val.shape[1] != X_pred.shape[1]:
        raise ValueError("All evaluation matrices must have the same gene dimension.")

    mean_real = X_target_val.mean(axis=0)
    mean_pred = X_pred.mean(axis=0)
    det_real = (X_target_val > threshold).mean(axis=0)
    det_pred = (X_pred > threshold).mean(axis=0)
    mean_r, mean_p = pearsonr(mean_real, mean_pred)
    det_r, det_p = pearsonr(det_real, det_pred)

    ks_df = per_gene_ks(X_target_val, X_pred, gene_names)
    wd_df = per_gene_wasserstein(X_target_val, X_pred, gene_names)

    lfc_real = historical_lfc_log1p_of_mean(X_source_val, X_target_val)
    lfc_pred = historical_lfc_log1p_of_mean(X_source_val, X_pred)
    lfc_r, lfc_p = pearsonr(lfc_real, lfc_pred)
    lfc_s = spearmanr(lfc_real, lfc_pred).correlation

    summary = {
        "gene_mean_pearson_r": float(mean_r),
        "gene_mean_pearson_p": float(mean_p),
        "detection_rate_pearson_r": float(det_r),
        "detection_rate_pearson_p": float(det_p),
        "median_per_gene_ks": float(ks_df.ks_stat.median()),
        "mean_per_gene_ks": float(ks_df.ks_stat.mean()),
        "fraction_ks_lt_0_1": float((ks_df.ks_stat < 0.1).mean()),
        "fraction_ks_lt_0_2": float((ks_df.ks_stat < 0.2).mean()),
        "median_per_gene_wasserstein": float(wd_df.wasserstein.median()),
        "mean_per_gene_wasserstein": float(wd_df.wasserstein.mean()),
        "historical_lfc_pearson_r": float(lfc_r),
        "historical_lfc_pearson_p": float(lfc_p),
        "historical_lfc_spearman_r": float(lfc_s),
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ks_df.sort_values("ks_stat").to_csv(out / "per_gene_ks.csv", index=False)
        wd_df.sort_values("wasserstein").to_csv(out / "per_gene_wasserstein.csv", index=False)
        np.savez_compressed(
            out / "effect_summaries.npz",
            lfc_historical_real=lfc_real,
            lfc_historical_pred=lfc_pred,
            log2fc_linear_real=linear_space_log2fc(X_source_val, X_target_val),
            log2fc_linear_pred=linear_space_log2fc(X_source_val, X_pred),
        )
        (out / "heldout_metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def final_comparison_metrics(X_ctrl_real, X_ctrl_pred, X_real, X_pred, top_de=5000, threshold=1e-6, ks_bad=0.2):
    """Metrics used in the final three-panel cross-model comparison."""
    ks = np.asarray([ks_2samp(X_real[:, g], X_pred[:, g]).statistic for g in range(X_real.shape[1])])
    det_real = (X_real > threshold).mean(axis=0)
    det_pred = (X_pred > threshold).mean(axis=0)
    r_det = pearsonr(det_real, det_pred)[0]

    effect_real = mean_log1p_shift(X_real, X_ctrl_real)
    effect_pred = mean_log1p_shift(X_pred, X_ctrl_pred)
    top_idx = np.argsort(np.abs(effect_real))[::-1][:top_de]
    r_lfc = pearsonr(effect_real[top_idx], effect_pred[top_idx])[0]
    return {
        "poorly_predicted_gene_count_KS_gt_0_2": int((ks > ks_bad).sum()),
        "poorly_predicted_gene_percent_KS_gt_0_2": float((ks > ks_bad).mean() * 100),
        "detection_rate_pearson_r": float(r_det),
        "logFC_top5000_DE_pearson_r": float(r_lfc),
        "mean_per_gene_ks": float(ks.mean()),
        "mean_per_gene_wasserstein": float(np.mean([
            wasserstein_distance(X_real[:, g], X_pred[:, g]) for g in range(X_real.shape[1])
        ])),
    }
