"""Evaluation metrics for cerebral organoid maturation translation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.stats import ks_2samp, pearsonr, wasserstein_distance


def _state_validation_rows(ae_npz: str, state: str, state_matrix: str) -> np.ndarray:
    z = np.load(ae_npz, allow_pickle=False)
    names = [str(x) for x in z["state_names"].tolist()]
    starts = z["slice_starts"].astype(int)
    ends = z["slice_ends"].astype(int)
    slices = {n: (s, e) for n, s, e in zip(names, starts, ends)}
    val_idx = z["val_idx"] if "val_idx" in z.files else z["test_idx"]
    s, e = slices[state]
    global_state = np.arange(s, e)
    global_val = np.intersect1d(global_state, val_idx)
    local_val = global_val - s
    x = np.load(state_matrix).astype(np.float32)
    return x[local_val]


def evaluate_translation(
    source_matrix: str,
    target_matrix: str,
    ae_npz: str,
    translated_npz: str,
    source_state: str,
    target_state: str,
    threshold: float = 1e-6,
    seed: int = 42,
) -> Dict[str, float]:
    translated = np.load(translated_npz, allow_pickle=False)
    if isinstance(translated, np.ndarray):
        x_fake = translated.astype(np.float32)
    else:
        x_fake = translated["X_translated"].astype(np.float32)
    x_real = _state_validation_rows(ae_npz, target_state, target_matrix)

    mean_corr = float(np.corrcoef(x_real.mean(0), x_fake.mean(0))[0, 1])
    det_corr = float(np.corrcoef((x_real > threshold).mean(0), (x_fake > threshold).mean(0))[0, 1])

    n = min(x_real.shape[0], x_fake.shape[0])
    rng = np.random.default_rng(seed)
    real_sub = x_real[rng.choice(x_real.shape[0], n, replace=False)]
    fake_sub = x_fake[rng.choice(x_fake.shape[0], n, replace=False)]

    ks = np.empty(x_real.shape[1], dtype=np.float32)
    w1 = np.empty(x_real.shape[1], dtype=np.float32)
    mean_diff = np.empty(x_real.shape[1], dtype=np.float32)
    for g in range(x_real.shape[1]):
        r = real_sub[:, g]
        f = fake_sub[:, g]
        ks[g] = ks_2samp(r, f).statistic
        w1[g] = wasserstein_distance(r, f)
        mean_diff[g] = float(f.mean() - r.mean())

    metrics = {
        "gene_mean_pearson_r": mean_corr,
        "detection_rate_pearson_r": det_corr,
        "median_ks": float(np.median(ks)),
        "median_wasserstein": float(np.median(w1)),
        "median_abs_mean_difference": float(np.median(np.abs(mean_diff))),
        "real_zero_fraction": float((x_real <= threshold).mean()),
        "translated_zero_fraction": float((x_fake <= threshold).mean()),
        "n_real_validation_cells": int(x_real.shape[0]),
        "n_translated_cells": int(x_fake.shape[0]),
    }

    # Historical mean-log1p effect recovery used in the primary notebook.
    x_source_all = np.load(source_matrix).astype(np.float32)
    real_effect = (x_real.mean(0) - x_source_all.mean(0)) / np.log(2.0)
    fake_effect = (x_fake.mean(0) - x_source_all.mean(0)) / np.log(2.0)
    metrics["historical_effect_pearson_r"] = float(pearsonr(real_effect, fake_effect).statistic)
    return metrics


def save_metrics(metrics: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def print_metrics(metrics: dict, label: str = "Organoid maturation") -> None:
    print(f"\n=== {label} validation ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
