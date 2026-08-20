#!/usr/bin/env python
"""Compare Cell Translator perturbation predictions with simple baselines.

This module reproduces the distribution-level benchmark implemented in the
source analysis while removing notebook-only fragments and machine-specific
paths. Inputs are supplied through a YAML configuration file.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp, pearsonr, spearmanr, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg


def load_array(path: str | Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    if method == "pearson":
        return float(pearsonr(x, y)[0])
    if method == "spearman":
        return float(spearmanr(x, y)[0])
    raise ValueError("method must be 'pearson' or 'spearman'")


def sample_rows(X, n, random_state=42):
    X = np.asarray(X)
    if X.shape[0] <= n:
        return X.copy()
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx]


def identity_baseline(X_source_eval):
    return X_source_eval.copy()


def mean_shift_baseline(X_source_train, X_target_train, X_source_eval):
    shift = X_target_train.mean(axis=0) - X_source_train.mean(axis=0)
    return X_source_eval + shift


def _fit_joint_pca(X_source_train, X_target_train, n_components, random_state):
    combined = np.vstack([X_source_train, X_target_train])
    scaler = StandardScaler()
    combined_std = scaler.fit_transform(combined)
    n_components = min(n_components, combined_std.shape[0] - 1, combined_std.shape[1])
    if n_components < 1:
        raise ValueError("PCA requires at least one effective component")
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(combined_std)
    return scaler, pca


def nearest_neighbor_baseline(X_source_train, X_target_train, X_source_eval,
                              n_components=50, random_state=42):
    scaler, pca = _fit_joint_pca(
        X_source_train, X_target_train, n_components, random_state
    )
    Z_target = pca.transform(scaler.transform(X_target_train))
    Z_source_eval = pca.transform(scaler.transform(X_source_eval))
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(Z_target)
    idx = nn.kneighbors(Z_source_eval, return_distance=False)[:, 0]
    return X_target_train[idx]


def linear_latent_shift_baseline(X_source_train, X_target_train, X_source_eval,
                                 n_components=50, random_state=42):
    scaler, pca = _fit_joint_pca(
        X_source_train, X_target_train, n_components, random_state
    )
    Z_source_train = pca.transform(scaler.transform(X_source_train))
    Z_target_train = pca.transform(scaler.transform(X_target_train))
    Z_source_eval = pca.transform(scaler.transform(X_source_eval))
    latent_shift = Z_target_train.mean(axis=0) - Z_source_train.mean(axis=0)
    return scaler.inverse_transform(pca.inverse_transform(Z_source_eval + latent_shift))


def discriminator_auc(X_pred, X_real, n_cells=1000, random_state=42):
    n = min(X_pred.shape[0], X_real.shape[0], n_cells)
    if n < 50:
        return np.nan
    pred = sample_rows(X_pred, n, random_state)
    real = sample_rows(X_real, n, random_state + 1)
    X = np.vstack([pred, real])
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    X_std = StandardScaler().fit_transform(X)
    n_components = min(30, X_std.shape[0] - 1, X_std.shape[1])
    X_pca = PCA(n_components=n_components, random_state=random_state).fit_transform(X_std)
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    return float(cross_val_score(clf, X_pca, y, cv=cv, scoring="roc_auc").mean())


def evaluate_prediction(perturbation, method, X_source_eval, X_target_eval, X_pred,
                        top_k=500, ks_max_genes=3000, auc_cells=1000,
                        random_state=42):
    X_source_eval = np.asarray(X_source_eval, dtype=np.float32)
    X_target_eval = np.asarray(X_target_eval, dtype=np.float32)
    X_pred = np.asarray(X_pred, dtype=np.float32)
    if not (X_source_eval.shape[1] == X_target_eval.shape[1] == X_pred.shape[1]):
        raise ValueError("Feature dimension mismatch")

    source_mean = X_source_eval.mean(axis=0)
    target_mean = X_target_eval.mean(axis=0)
    pred_mean = X_pred.mean(axis=0)
    real_effect = target_mean - source_mean
    pred_effect = pred_mean - source_mean

    effect_pearson = safe_corr(real_effect, pred_effect, "pearson")
    effect_spearman = safe_corr(real_effect, pred_effect, "spearman")
    target_mean_pearson = safe_corr(target_mean, pred_mean, "pearson")
    mean_abs_mean_error = float(np.mean(np.abs(pred_mean - target_mean)))
    mean_rmse = float(np.sqrt(np.mean((pred_mean - target_mean) ** 2)))

    top_k = min(top_k, len(real_effect))
    top_real_idx = np.argsort(np.abs(real_effect))[-top_k:]
    top_pred_idx = np.argsort(np.abs(pred_effect))[-top_k:]
    sign_agreement = float(np.mean(
        np.sign(real_effect[top_real_idx]) == np.sign(pred_effect[top_real_idx])
    ))
    union = set(top_real_idx) | set(top_pred_idx)
    jaccard = float(len(set(top_real_idx) & set(top_pred_idx)) / len(union)) if union else np.nan

    if ks_max_genes is None or ks_max_genes >= X_target_eval.shape[1]:
        dist_idx = np.arange(X_target_eval.shape[1])
    else:
        dist_idx = np.argsort(np.abs(real_effect))[-ks_max_genes:]
    ks_values = [ks_2samp(X_pred[:, j], X_target_eval[:, j]).statistic for j in dist_idx]
    w_values = [wasserstein_distance(X_pred[:, j], X_target_eval[:, j]) for j in dist_idx]

    return {
        "perturbation": perturbation,
        "method": method,
        "n_source_eval": int(X_source_eval.shape[0]),
        "n_target_eval": int(X_target_eval.shape[0]),
        "n_pred": int(X_pred.shape[0]),
        "n_genes": int(X_pred.shape[1]),
        "effect_pearson": effect_pearson,
        "effect_spearman": effect_spearman,
        "target_mean_pearson": target_mean_pearson,
        "sign_agreement_top500": sign_agreement,
        "top500_jaccard": jaccard,
        "mean_abs_mean_error": mean_abs_mean_error,
        "mean_rmse": mean_rmse,
        "median_gene_ks": float(np.median(ks_values)),
        "mean_gene_ks": float(np.mean(ks_values)),
        "mean_wasserstein": float(np.mean(w_values)),
        "pred_vs_real_auc": discriminator_auc(
            X_pred, X_target_eval, n_cells=auc_cells, random_state=random_state
        ),
    }


def run_benchmark(cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    settings = cfg["settings"]
    seed = int(settings.get("random_state", 42))
    test_size = float(settings.get("test_size", 0.30))
    pca_components = int(settings.get("pca_n_components", 50))
    top_k = int(settings.get("top_k_genes", 500))
    ks_max = settings.get("ks_max_genes", 3000)
    ks_max = None if ks_max is None else int(ks_max)
    auc_cells = int(settings.get("auc_cells", 1000))
    clip_nonnegative = bool(settings.get("clip_nonnegative", True))

    X_control_all = load_array(cfg["data"]["control"]).astype(np.float32)
    gene_names = load_array(cfg["data"]["gene_names"])
    if len(gene_names) != X_control_all.shape[1]:
        raise ValueError("gene_names length does not match control matrix columns")

    rows = []
    for name, entry in cfg["perturbations"].items():
        X_target_all = load_array(entry["measured"]).astype(np.float32)
        X_model_pred_all = load_array(entry["cell_translator_prediction"]).astype(np.float32)
        if not (X_control_all.shape[1] == X_target_all.shape[1] == X_model_pred_all.shape[1]):
            raise ValueError(f"Gene dimension mismatch for {name}")

        src_tr, src_ev = train_test_split(
            np.arange(len(X_control_all)), test_size=test_size,
            random_state=seed, shuffle=True
        )
        tgt_tr, tgt_ev = train_test_split(
            np.arange(len(X_target_all)), test_size=test_size,
            random_state=seed, shuffle=True
        )
        X_source_train, X_source_eval = X_control_all[src_tr], X_control_all[src_ev]
        X_target_train, X_target_eval = X_target_all[tgt_tr], X_target_all[tgt_ev]

        n_eval = min(len(X_source_eval), len(X_model_pred_all))
        X_source_eval = sample_rows(X_source_eval, n_eval, seed)
        X_model_pred = sample_rows(X_model_pred_all, n_eval, seed)

        predictions: Dict[str, np.ndarray] = {
            "Identity": identity_baseline(X_source_eval),
            "MeanShift": mean_shift_baseline(X_source_train, X_target_train, X_source_eval),
            "NearestNeighbor": nearest_neighbor_baseline(
                X_source_train, X_target_train, X_source_eval, pca_components, seed
            ),
            "LinearLatentShift": linear_latent_shift_baseline(
                X_source_train, X_target_train, X_source_eval, pca_components, seed
            ),
            "CellTranslator": X_model_pred,
        }

        reference_nonnegative = (
            np.nanmin(X_control_all) >= 0 and np.nanmin(X_target_all) >= 0
            and np.nanmin(X_model_pred_all) >= 0
        )
        if clip_nonnegative and reference_nonnegative:
            predictions = {k: np.clip(v, 0, None) for k, v in predictions.items()}

        for method, X_pred in predictions.items():
            result = evaluate_prediction(
                name, method, X_source_eval, X_target_eval, X_pred,
                top_k=top_k, ks_max_genes=ks_max, auc_cells=auc_cells,
                random_state=seed,
            )
            rows.append(result)

    raw = pd.DataFrame(rows)
    clean = raw.copy()
    # Identity has no perturbation effect; display these effect metrics as NA,
    # matching the final plotting treatment in the source analysis.
    clean.loc[clean["method"].eq("Identity"), [
        "effect_pearson", "sign_agreement_top500"
    ]] = np.nan

    summary = (
        clean.groupby("method", as_index=False)
        .agg({
            "effect_pearson": "mean", "effect_spearman": "mean",
            "target_mean_pearson": "mean", "sign_agreement_top500": "mean",
            "top500_jaccard": "mean", "mean_abs_mean_error": "mean",
            "mean_rmse": "mean", "median_gene_ks": "mean",
            "mean_gene_ks": "mean", "mean_wasserstein": "mean",
            "pred_vs_real_auc": "mean",
        })
    )
    return raw, clean, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    raw, clean, summary = run_benchmark(cfg)
    raw.to_csv(outdir / "baseline_comparison_all_metrics_raw.csv", index=False)
    clean.to_csv(outdir / "baseline_comparison_all_metrics.csv", index=False)
    summary.to_csv(outdir / "baseline_comparison_mean_by_method.csv", index=False)
    print(clean.to_string(index=False))
    print(f"\nSaved benchmark outputs to: {outdir}")


if __name__ == "__main__":
    main()
