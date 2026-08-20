"""Inverse-transform CellOT PCA predictions and align them to the full gene space."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from .common import load_yaml, to_dense_float32, write_id_file


def reconstruct_full_gene_space(
    x_hvg_pred: np.ndarray,
    hvg_genes: np.ndarray,
    gene_names: np.ndarray,
    control_mean: np.ndarray,
) -> np.ndarray:
    gene_names = np.asarray(gene_names).astype(str)
    hvg_genes = np.asarray(hvg_genes).astype(str)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    pred_cols, full_cols = [], []
    for i, gene in enumerate(hvg_genes):
        if gene in gene_to_idx:
            pred_cols.append(i)
            full_cols.append(gene_to_idx[gene])
    if not full_cols:
        raise ValueError("No CellOT HVGs overlap the reference gene list")
    out = np.tile(np.asarray(control_mean, dtype=np.float32), (x_hvg_pred.shape[0], 1))
    out[:, np.asarray(full_cols)] = x_hvg_pred[:, np.asarray(pred_cols)]
    return out.astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    ccfg = cfg["cellot"]

    import anndata as ad

    outroot = Path(ccfg["output_root"])
    with (outroot / "pca_ctrl_hvg.pkl").open("rb") as handle:
        pdata = pickle.load(handle)
    pca = pdata["pca"]
    hvg_genes = np.asarray(pdata["hvg_genes"]).astype(str)

    gene_names = np.load(cfg["dataset"]["gene_names"], allow_pickle=True).astype(str)
    control = np.load(cfg["dataset"]["control_matrix"]).astype(np.float32)
    control_mean = control.mean(axis=0)

    common_ids = None
    for gene in cfg["perturbations"]:
        imputed = outroot / gene / "model-cellot-pca50" / "evals_iid_data_space" / "imputed.h5ad"
        if not imputed.exists():
            raise FileNotFoundError(imputed)
        adata_imp = ad.read_h5ad(imputed)
        ids = list(map(str, adata_imp.obs_names))
        if common_ids is None:
            common_ids = ids
        elif ids != common_ids:
            raise ValueError(f"CellOT held-out control IDs differ for {gene}")

        x_pca = to_dense_float32(adata_imp.X)
        if bool(ccfg.get("recover_expm1_like_pca_output", True)) and x_pca.min() > 0 and x_pca.max() > 100:
            print(f"[{gene}] warning: PCA prediction scale looks exponentiated; applying log1p recovery")
            x_pca = np.log1p(x_pca)

        x_hvg = pca.inverse_transform(x_pca).astype(np.float32)
        if bool(ccfg.get("clip_gene_space_nonnegative", True)):
            x_hvg = np.clip(x_hvg, 0.0, None)
        x_full = reconstruct_full_gene_space(x_hvg, hvg_genes, gene_names, control_mean)

        outdir = outroot / gene / "model-cellot-pca50"
        outpath = outdir / f"cellot_pred_{gene}_pca50_fullgene.npy"
        np.save(outpath, x_full)

        target_path = cfg.get("targets", {}).get(gene)
        if target_path:
            x_real = np.load(target_path).astype(np.float32)
            r = pearsonr(x_real.mean(axis=0), x_full.mean(axis=0)).statistic
            print(f"[{gene}] gene-mean Pearson r vs measured target: {r:.6f}")
        print(f"[{gene}] saved {outpath} shape={x_full.shape}")

    ids_path = Path(cfg["common_test_control_ids"])
    write_id_file(common_ids or [], ids_path)
    expected = int(cfg["dataset"].get("expected_test_control_cells", len(common_ids or [])))
    if len(common_ids or []) != expected:
        raise ValueError(f"CellOT held-out control count {len(common_ids or [])} != expected {expected}")
    print(f"Saved common held-out control IDs: {ids_path} ({len(common_ids or [])} rows)")


if __name__ == "__main__":
    main()
