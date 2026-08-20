"""Prepare the GSE132188 NGN3-high-EP to Endocrine translation matrices.

The source atlas is already normalized/log1p. The
reported model was trained after selecting 5,000 Seurat-flavor HVGs on the
combined source/target subset and then force-including three additional
endocrine marker genes that were not among those HVGs, yielding 5,003 genes.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from config_utils import load_config


def _dense_float32(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D expression matrix, received {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("Prepared expression matrix contains NaN/Inf values.")
    if np.any(x < 0):
        raise ValueError("HurdleAE inputs must be non-negative.")
    return x


def _numeric_day(series: pd.Series) -> pd.Series:
    # The source notebook accepted numeric values and string representations.
    extracted = series.astype(str).str.extract(r"(\d+\.?\d*)")[0]
    values = pd.to_numeric(extracted, errors="coerce")
    if values.isna().any():
        raise ValueError(f"{int(values.isna().sum())} cells have an unparseable day value.")
    return values


def prepare_dataset(cfg: dict) -> dict[str, str]:
    prep = cfg["preparation"]
    atlas_path = prep["input_h5ad"]
    adata = ad.read_h5ad(atlas_path)

    cluster_col = prep["cluster_column"]
    day_col = prep["day_column"]
    for col in (cluster_col, day_col):
        if col not in adata.obs:
            raise KeyError(f"Required obs column {col!r} is absent from the input atlas.")

    day = _numeric_day(adata.obs[day_col])
    source_mask = (
        (adata.obs[cluster_col].astype(str) == prep["source_cluster_label"])
        & (day <= float(prep["source_day_max"]))
    )
    target_mask = (
        (adata.obs[cluster_col].astype(str) == prep["target_cluster_label"])
        & (day >= float(prep["target_day_min"]))
    )

    n_source = int(source_mask.sum())
    n_target = int(target_mask.sum())
    print(f"{cfg['source_state']} cells: {n_source}")
    print(f"{cfg['target_state']} cells: {n_target}")

    expected = prep.get("expected_cell_counts", {})
    if expected:
        if n_source != int(expected.get("source", n_source)):
            raise RuntimeError(f"Source cell count mismatch: observed {n_source}, expected {expected['source']}.")
        if n_target != int(expected.get("target", n_target)):
            raise RuntimeError(f"Target cell count mismatch: observed {n_target}, expected {expected['target']}.")

    # Final executed notebook path: HVGs are recomputed only on the source +
    # target subset. No additional normalization/log1p is performed.
    subset = adata[source_mask | target_mask].copy()
    sc.pp.highly_variable_genes(
        subset,
        n_top_genes=int(prep["n_hvg_base"]),
        flavor=prep["hvg_flavor"],
        subset=False,
    )
    hvg = subset.var_names[subset.var["highly_variable"].to_numpy()].astype(str).to_numpy()
    if len(hvg) != int(prep["n_hvg_base"]):
        raise RuntimeError(f"Expected {prep['n_hvg_base']} base HVGs; obtained {len(hvg)}.")

    force_markers = [str(g) for g in prep["force_include_markers"]]
    atlas_genes = set(map(str, subset.var_names))
    hvg_set = set(map(str, hvg))
    extras = [g for g in force_markers if g in atlas_genes and g not in hvg_set]
    genes = np.concatenate([hvg.astype(object), np.asarray(extras, dtype=object)]).astype(str)

    print(f"Base HVGs: {len(hvg)}")
    print(f"Force-added markers ({len(extras)}): {extras}")
    print(f"Final genes: {len(genes)}")

    expected_extras = prep.get("expected_force_added_markers")
    if expected_extras is not None and list(extras) != list(expected_extras):
        raise RuntimeError(
            f"Force-added marker set differs from the reported analysis: {extras} != {expected_extras}"
        )
    if len(genes) != int(prep["expected_final_gene_count"]):
        raise RuntimeError(
            f"Final gene count mismatch: {len(genes)} != {prep['expected_final_gene_count']}"
        )

    # Preserve original atlas row order within each state.
    x_source = _dense_float32(adata[source_mask, genes].X)
    x_target = _dense_float32(adata[target_mask, genes].X)

    out = Path(cfg["paths"]["prepared_dir"])
    out.mkdir(parents=True, exist_ok=True)
    source_path = out / f"{cfg['source_state']}_X_hvg.npy"
    target_path = out / f"{cfg['target_state']}_X_hvg.npy"
    genes_npy = out / "hvg_genes.npy"
    genes_csv = out / "hvg_genes.csv"

    np.save(source_path, x_source)
    np.save(target_path, x_target)
    np.save(genes_npy, genes)
    pd.Series(genes, name="gene").to_csv(genes_csv, index=False)

    print(f"Saved {cfg['source_state']}: {x_source.shape}")
    print(f"Saved {cfg['target_state']}: {x_target.shape}")
    return {
        cfg["source_state"]: str(source_path),
        cfg["target_state"]: str(target_path),
        "hvg_genes": str(genes_npy),
        "hvg_genes_csv": str(genes_csv),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare_dataset(load_config(args.config))


if __name__ == "__main__":
    main()
