"""Core held-out translation evaluation for Cell Painting morphology."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, ks_2samp
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _select_equal(X_target, X_translated, seed):
    n = min(len(X_target), len(X_translated))
    rng = np.random.default_rng(seed)
    target_idx = rng.choice(len(X_target), n, replace=False)
    translated_idx = rng.choice(len(X_translated), n, replace=False)
    return X_target[target_idx], X_translated[translated_idx]


def _target_neighbor_percent(
    X_source_validation,
    X_target_validation,
    X_translated,
    method,
    X_source_train=None,
    X_target_train=None,
):
    method = str(method)
    if method == "train_common_scaled_5nn":
        if X_source_train is None or X_target_train is None:
            raise ValueError("Training matrices are required for train_common_scaled_5nn.")
        X_train = np.vstack([X_source_train, X_target_train])
        y_train = np.asarray(["source"] * len(X_source_train) + ["target"] * len(X_target_train))
        scaler = StandardScaler().fit(X_train)
        clf = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(X_train), y_train)
        return float((clf.predict(scaler.transform(X_translated)) == "target").mean() * 100.0)

    if method == "validation_unscaled_5nn":
        X_ref = np.vstack([X_target_validation, X_source_validation])
        y_ref = np.asarray(["target"] * len(X_target_validation) + ["source"] * len(X_source_validation))
        clf = KNeighborsClassifier(n_neighbors=5).fit(X_ref, y_ref)
        return float((clf.predict(X_translated) == "target").mean() * 100.0)

    if method == "validation_unscaled_1nn":
        X_ref = np.vstack([X_source_validation, X_target_validation])
        labels = np.asarray(["source"] * len(X_source_validation) + ["target"] * len(X_target_validation))
        nn = NearestNeighbors(n_neighbors=1).fit(X_ref)
        _, idx = nn.kneighbors(X_translated)
        return float((labels[idx.ravel()] == "target").mean() * 100.0)

    raise ValueError(f"Unknown nn_method: {method}")


def evaluate_translation(
    X_source_validation,
    X_target_validation,
    X_translated,
    *,
    seed=42,
    correlation_sampling="equal",
    ks_sampling="equal",
    std_ddof=0,
    nan_aware=False,
    nn_method="validation_unscaled_1nn",
    X_source_train=None,
    X_target_train=None,
):
    """Evaluate a transition using the exact workflow-specific notebook conventions."""
    X_source_validation = np.asarray(X_source_validation, dtype=np.float32)
    X_target_validation = np.asarray(X_target_validation, dtype=np.float32)
    X_translated = np.asarray(X_translated, dtype=np.float32)

    if correlation_sampling == "equal":
        target_corr, translated_corr = _select_equal(X_target_validation, X_translated, seed)
    elif correlation_sampling == "full":
        target_corr, translated_corr = X_target_validation, X_translated
    else:
        raise ValueError("correlation_sampling must be 'equal' or 'full'.")

    mean_fn = np.nanmean if nan_aware else np.mean
    std_fn = np.nanstd if nan_aware else np.std
    mean_target = mean_fn(target_corr, axis=0)
    mean_translated = mean_fn(translated_corr, axis=0)
    std_target = std_fn(target_corr, axis=0, ddof=std_ddof)
    std_translated = std_fn(translated_corr, axis=0, ddof=std_ddof)
    valid_mean = np.isfinite(mean_target) & np.isfinite(mean_translated)
    valid_std = np.isfinite(std_target) & np.isfinite(std_translated)
    r_mean = pearsonr(mean_target[valid_mean], mean_translated[valid_mean])[0]
    r_std = pearsonr(std_target[valid_std], std_translated[valid_std])[0]

    if ks_sampling == "equal":
        target_ks, translated_ks = _select_equal(X_target_validation, X_translated, seed)
    elif ks_sampling == "full":
        target_ks, translated_ks = X_target_validation, X_translated
    else:
        raise ValueError("ks_sampling must be 'equal' or 'full'.")

    ks = np.asarray(
        [ks_2samp(target_ks[:, j], translated_ks[:, j]).statistic for j in range(target_ks.shape[1])],
        dtype=np.float32,
    )

    target_nn_percent = _target_neighbor_percent(
        X_source_validation,
        X_target_validation,
        X_translated,
        nn_method,
        X_source_train=X_source_train,
        X_target_train=X_target_train,
    )

    return {
        "feature_mean_pearson_r": float(r_mean),
        "feature_std_pearson_r": float(r_std),
        "median_ks": float(np.nanmedian(ks)),
        "mean_ks": float(np.nanmean(ks)),
        "target_nn_percent": target_nn_percent,
        "n_compared": int(min(len(X_target_validation), len(X_translated))),
    }, ks


def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
