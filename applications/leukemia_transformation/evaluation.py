"""Core validation metrics for acute-leukemia Cell Translator workflows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, pearsonr, wasserstein_distance

from config_utils import load_config


def _safe_pearson(a, b) -> float:
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def _per_dimension_distribution(real: np.ndarray, fake: np.ndarray):
    ks = np.array([ks_2samp(real[:, j], fake[:, j]).statistic for j in range(real.shape[1])])
    w1 = np.array([wasserstein_distance(real[:, j], fake[:, j]) for j in range(real.shape[1])])
    mean_diff = fake.mean(0) - real.mean(0)
    return float(np.median(ks)), float(np.median(w1)), float(np.median(np.abs(mean_diff)))


def latent_metrics(source_z: np.ndarray, target_z: np.ndarray, fake_z: np.ndarray) -> dict:
    mean_real = target_z.mean(0); mean_fake = fake_z.mean(0)
    std_real = target_z.std(0, ddof=0); std_fake = fake_z.std(0, ddof=0)
    median_ks, median_w1, median_md = _per_dimension_distribution(target_z, fake_z)

    # Real and model-implied translation directions in standardized latent space.
    src_mean = source_z.mean(0)
    real_direction = mean_real - src_mean
    model_direction = mean_fake - src_mean
    denom = np.linalg.norm(real_direction) * np.linalg.norm(model_direction)
    cosine = float(np.dot(real_direction, model_direction) / denom) if denom > 0 else float("nan")

    return {
        "latent_mean_pearson_r": _safe_pearson(mean_real, mean_fake),
        "latent_mean_l2": float(np.linalg.norm(mean_real - mean_fake)),
        "latent_mean_mae": float(np.mean(np.abs(mean_real - mean_fake))),
        "latent_std_pearson_r": _safe_pearson(std_real, std_fake),
        "latent_std_l2": float(np.linalg.norm(std_real - std_fake)),
        "latent_std_mae": float(np.mean(np.abs(std_real - std_fake))),
        "latent_std_relative_mae": float(np.mean(np.abs(std_real - std_fake) / (std_real + 1e-8))),
        "median_latent_ks": median_ks,
        "median_latent_wasserstein": median_w1,
        "median_abs_latent_mean_difference": median_md,
        "direction_cosine_similarity": cosine,
    }


def _sample_equal(real: np.ndarray, fake: np.ndarray, seed: int | None):
    n = min(real.shape[0], fake.shape[0])
    if seed is None:
        ir = np.random.choice(real.shape[0], n, replace=False)
        iff = np.random.choice(fake.shape[0], n, replace=False)
    else:
        rng = np.random.default_rng(seed)
        ir = rng.choice(real.shape[0], n, replace=False)
        iff = rng.choice(fake.shape[0], n, replace=False)
    return real[ir], fake[iff]


def gene_distribution_metrics(real: np.ndarray, fake: np.ndarray, seed: int | None = None) -> dict:
    real, fake = _sample_equal(real, fake, seed)
    n_genes = real.shape[1]
    ks = np.empty(n_genes, dtype=np.float32)
    w1 = np.empty(n_genes, dtype=np.float32)
    md = np.empty(n_genes, dtype=np.float32)
    for j in range(n_genes):
        ks[j] = ks_2samp(real[:, j], fake[:, j]).statistic
        w1[j] = wasserstein_distance(real[:, j], fake[:, j])
        md[j] = fake[:, j].mean() - real[:, j].mean()
    return {
        "median_gene_ks": float(np.median(ks)),
        "median_gene_wasserstein": float(np.median(w1)),
        "median_abs_gene_mean_difference": float(np.median(np.abs(md))),
        "fraction_gene_ks_lt_0_05": float(np.mean(ks < 0.05)),
        "fraction_gene_ks_gt_0_50": float(np.mean(ks > 0.50)),
        "fraction_gene_ks_gt_0_90": float(np.mean(ks > 0.90)),
    }


def log2fc_metrics(source: np.ndarray, target: np.ndarray, fake_expected: np.ndarray, genes: np.ndarray,
                   expression_threshold: float = 0.01, pseudocount: float = 0.1,
                   linear_clip_max: float = 20.0) -> tuple[dict, dict[str, np.ndarray]]:
    def mean_linear(x):
        return np.expm1(np.clip(np.asarray(x, np.float32), None, linear_clip_max)).mean(0)
    src = mean_linear(source); tgt = mean_linear(target); fake = mean_linear(fake_expected)
    real_fc = np.log2((tgt + pseudocount) / (src + pseudocount))
    fake_fc = np.log2((fake + pseudocount) / (src + pseudocount))
    finite = np.isfinite(real_fc) & np.isfinite(fake_fc)
    # Historical final notebook filter used source OR measured-target mean
    # expression; translated expression was not part of the keep criterion.
    keep = finite & ((src > expression_threshold) | (tgt > expression_threshold))
    return {
        "log2fc_pearson_r": _safe_pearson(real_fc[keep], fake_fc[keep]),
        "log2fc_genes_retained": int(keep.sum()),
    }, {
        "gene": np.asarray(genes).astype(str)[keep],
        "log2fc_real": real_fc[keep],
        "log2fc_fake": fake_fc[keep],
    }


def evaluate(config: dict, translated_npz: str) -> dict:
    src = np.load(config["paths"]["source_expression"]).astype(np.float32)
    tgt = np.load(config["paths"]["target_expression"]).astype(np.float32)
    genes = np.load(config["paths"]["gene_names"], allow_pickle=False).astype(str)
    tr = np.load(translated_npz, allow_pickle=False)

    output_key = config["evaluation"]["distribution_output"]
    output_map = {
        "expected": "X_expected",
        "hard": "X_hard",
        "library_matched": "X_library_matched",
    }
    if output_key not in output_map:
        raise ValueError(f"Unknown distribution_output={output_key!r}")
    fake_distribution = tr[output_map[output_key]].astype(np.float32)

    metrics = {
        "workflow": config["workflow"],
        "n_source_cells": int(src.shape[0]),
        "n_target_cells": int(tgt.shape[0]),
        "n_source_validation_cells": int(tr["source_z"].shape[0]),
        "n_target_validation_cells": int(tr["target_z"].shape[0]),
        "n_translated_cells": int(fake_distribution.shape[0]),
        "target_detection_fraction": float((tgt > 0).mean()),
        "hard_detection_fraction": float((tr["X_hard"] > 0).mean()),
        "p_threshold": float(tr["p_threshold"]),
    }
    metrics.update(latent_metrics(tr["source_z"], tr["target_z"], tr["fake_z"]))
    metrics.update(gene_distribution_metrics(
        tgt, fake_distribution, config["evaluation"].get("sampling_seed")
    ))
    lfc, arrays = log2fc_metrics(
        src, tgt, tr["X_expected"], genes,
        expression_threshold=float(config["log2fc"]["expression_threshold"]),
        pseudocount=float(config["log2fc"]["pseudocount"]),
        linear_clip_max=float(config["log2fc"]["linear_clip_max"]),
    )
    metrics.update(lfc)

    if config["evaluation"].get("also_evaluate_library_matched", False):
        alt = gene_distribution_metrics(
            tgt, tr["X_library_matched"], config["evaluation"].get("sampling_seed")
        )
        metrics["library_matched_distribution"] = alt

    return metrics, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--translated", required=True)
    ap.add_argument("--output")
    args = ap.parse_args()
    cfg = load_config(args.config)
    metrics, arrays = evaluate(cfg, args.translated)
    out = Path(args.output or (Path(cfg["paths"]["output_dir"]) / "evaluation_metrics.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(out.with_name("log2fc_arrays.npz"), **arrays)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {out}")


if __name__ == "__main__":
    main()
