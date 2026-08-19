"""Dataset preparation for the three cerebral organoid maturation datasets."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from config_utils import load_config


def _feature_names(table_path: str) -> np.ndarray:
    table = pd.read_csv(table_path, header=None)
    col = 1 if table.shape[1] > 1 else 0
    return table.iloc[:, col].astype(str).to_numpy()


def _read_primary_mtx_series(cfg: dict):
    adatas = []
    labels = []
    for label, paths in cfg["preparation"]["timepoints"].items():
        adata = sc.read_mtx(paths["expression"]).T
        barcodes = pd.read_csv(paths["barcodes"], header=None).iloc[:, 0].astype(str).to_numpy()
        genes = _feature_names(paths["genes"])
        if adata.n_obs != len(barcodes) or adata.n_vars != len(genes):
            raise ValueError(f"Shape/metadata mismatch for {label}: matrix={adata.shape}, barcodes={len(barcodes)}, genes={len(genes)}")
        adata.obs_names = barcodes
        adata.var_names = genes
        adata.obs["timepoint"] = label
        adatas.append(adata)
        labels.append(label)

    # The source notebook used AnnData.concatenate (inner gene intersection).
    combined = adatas[0].concatenate(
        *adatas[1:],
        batch_key="timepoint",
        batch_categories=labels,
    )
    return combined


def _read_h5ad_age_dataset(cfg: dict):
    prep = cfg["preparation"]
    adata = sc.read_h5ad(prep["input_h5ad"])
    age_column = prep["age_column"]
    ages = pd.to_numeric(adata.obs[age_column], errors="coerce")
    if ages.isna().any():
        raise ValueError(f"{int(ages.isna().sum())} cells have non-numeric values in {age_column!r}.")
    adata.obs[age_column] = ages

    pieces = []
    for age in prep["selected_ages_days"]:
        sub = adata[adata.obs[age_column] == float(age)].copy()
        label = f"{int(age)}d"
        if sub.n_obs == 0:
            raise ValueError(f"No cells found for exact age {age} days.")
        sub.obs["timepoint"] = label
        pieces.append(sub)
        print(f"{label}: {sub.n_obs} cells")

    return sc.concat(pieces, join="outer", label=None, index_unique=None)


def _dense_float32(x):
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Prepared expression matrix contains NaN/Inf values.")
    if np.any(x < 0):
        raise ValueError("HurdleAE inputs must be non-negative.")
    return x


def prepare_dataset(cfg: dict) -> Dict[str, str]:
    prep = cfg["preparation"]
    if prep["input_type"] == "mtx_series":
        adata = _read_primary_mtx_series(cfg)
    elif prep["input_type"] == "h5ad_age":
        adata = _read_h5ad_age_dataset(cfg)
    else:
        raise ValueError(f"Unsupported input_type: {prep['input_type']}")

    print(f"Combined dataset: {adata.n_obs} cells x {adata.n_vars} genes")
    # Inputs in all three source notebooks were already normalized/log1p.
    # No additional normalization or QC filtering was applied before HVG selection.
    sc.pp.highly_variable_genes(
        adata,
        flavor=prep["hvg_flavor"],
        n_top_genes=int(prep["n_hvg"]),
        batch_key="timepoint",
    )
    hvg_mask = adata.var["highly_variable"].to_numpy()
    if int(hvg_mask.sum()) != int(prep["n_hvg"]):
        raise RuntimeError(f"Expected {prep['n_hvg']} HVGs; obtained {int(hvg_mask.sum())}.")

    out_dir = Path(cfg["paths"]["prepared_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    hvg_genes = adata.var_names[hvg_mask].astype(str).to_numpy()
    gene_path = out_dir / "hvg_genes.npy"
    np.save(gene_path, hvg_genes)

    outputs = {"hvg_genes": str(gene_path)}
    for tp in adata.obs["timepoint"].astype(str).unique():
        sub = adata[adata.obs["timepoint"].astype(str) == tp, hvg_mask]
        x = _dense_float32(sub.X)
        path = out_dir / f"{tp}_X_hvg.npy"
        np.save(path, x)
        outputs[tp] = str(path)
        print(f"Saved {tp}: {x.shape} -> {path}")

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    prepare_dataset(load_config(args.config))


if __name__ == "__main__":
    main()
