"""Postprocess scGPT predictions for cross-method benchmark evaluation.

The exported scGPT prediction matrices are mapped to the common gene order,
converted to the absolute-expression representation used in the benchmark by
adding the measured control-state mean expression, and clipped at zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .common import (
    delta_to_absolute_control_mean,
    load_yaml,
    map_to_reference_gene_order,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    scfg = cfg["scgpt"]

    reference_genes = np.load(cfg["dataset"]["gene_names"], allow_pickle=True).astype(str)
    control = np.load(cfg["dataset"]["control_matrix"]).astype(np.float32)
    expected_cells = int(cfg["dataset"]["expected_test_control_cells"])
    expected_genes = int(cfg["dataset"]["expected_shape"][1])

    outroot = Path(scfg["output_root"])
    outroot.mkdir(parents=True, exist_ok=True)
    for gene in cfg["perturbations"]:
        raw_path = Path(scfg["raw_npz"][gene])
        npz = np.load(raw_path, allow_pickle=True)
        if "yhat" not in npz.files or "gene_names" not in npz.files:
            raise KeyError(f"{raw_path} must contain yhat and gene_names")
        yhat = np.asarray(npz["yhat"], dtype=np.float32)
        source_genes = np.asarray(npz["gene_names"]).astype(str)
        yhat = map_to_reference_gene_order(yhat, source_genes, reference_genes)
        if yhat.shape != (expected_cells, expected_genes):
            raise ValueError(f"{gene}: raw scGPT shape {yhat.shape}; expected {(expected_cells, expected_genes)}")

        x_abs = delta_to_absolute_control_mean(
            yhat,
            control,
            clip_nonnegative=bool(scfg.get("clip_nonnegative", True)),
        )
        outpath = outroot / f"scgpt_pred_{gene}_absolute.npy"
        np.save(outpath, x_abs)

        target_path = cfg.get("targets", {}).get(gene)
        if target_path:
            x_real = np.load(target_path).astype(np.float32)
            mean_real = x_real.mean(axis=0)
            mean_pred = x_abs.mean(axis=0)
            r = pearsonr(mean_real, mean_pred).statistic
            rho = spearmanr(mean_real, mean_pred).statistic
            det_real = (x_real > float(scfg.get("detection_threshold", 1e-6))).mean(axis=0)
            det_pred = (x_abs > float(scfg.get("detection_threshold", 1e-6))).mean(axis=0)
            det_r = pearsonr(det_real, det_pred).statistic
            print(f"[{gene}] mean Pearson={r:.6f} Spearman={rho:.6f} detection_r={det_r:.6f}")
        print(f"[{gene}] saved {outpath} shape={x_abs.shape}")


if __name__ == "__main__":
    main()
