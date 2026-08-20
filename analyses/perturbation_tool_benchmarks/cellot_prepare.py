"""Prepare the PCA(50) CellOT benchmark input used in the reported comparison."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from .common import load_yaml, to_dense_float32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)

    import anndata as ad
    import scanpy as sc
    from sklearn.decomposition import PCA

    dataset = cfg["dataset"]
    ccfg = cfg["cellot"]
    outroot = Path(ccfg["output_root"])
    outroot.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(dataset["h5ad"])
    condition_key = dataset.get("condition_key", "condition")
    control_label = dataset.get("control_label", "ctrl")

    expected_shape = tuple(dataset.get("expected_shape", []))
    if expected_shape and adata.shape != expected_shape:
        raise ValueError(f"AnnData shape {adata.shape} != expected {expected_shape}")

    control_mask = np.asarray(adata.obs[condition_key] == control_label)
    adata_ctrl = adata[control_mask].copy()
    sc.pp.highly_variable_genes(
        adata_ctrl,
        n_top_genes=int(ccfg["n_hvg"]),
        flavor=str(ccfg["hvg_flavor"]),
        subset=False,
    )
    hvg_mask = np.asarray(adata_ctrl.var["highly_variable"])
    hvg_genes = np.asarray(adata.var_names[hvg_mask]).astype(str)
    if len(hvg_genes) != int(ccfg["n_hvg"]):
        raise ValueError(f"Selected {len(hvg_genes)} HVGs; expected {ccfg['n_hvg']}")

    adata_hvg = adata[:, hvg_genes].copy()
    x_all = to_dense_float32(adata_hvg.X)
    x_ctrl = x_all[control_mask]

    pca = PCA(
        n_components=int(ccfg["n_pcs"]),
        random_state=int(ccfg["pca_random_state"]),
    )
    pca.fit(x_ctrl)
    x_pca = pca.transform(x_all).astype(np.float32)

    pca_file = outroot / "pca_ctrl_hvg.pkl"
    with pca_file.open("wb") as handle:
        pickle.dump({"pca": pca, "hvg_genes": hvg_genes.tolist()}, handle)

    adata_pca = ad.AnnData(X=x_pca)
    adata_pca.obs = adata.obs.copy()
    adata_pca.var_names = [f"PC{i + 1}" for i in range(x_pca.shape[1])]
    adata_pca.obs_names_make_unique()
    pca_h5ad = outroot / "neuron_allconds_pca50.h5ad"
    adata_pca.write_h5ad(pca_h5ad)

    task_file = outroot / "task_pca50.yaml"
    task_text = f"""data:\n  type: cell\n  path: {pca_h5ad}\n  condition: {condition_key}\n  source: {control_label}\n  target: null\ndataloader:\n  batch_size: {int(ccfg['batch_size'])}\n  shuffle: true\ndatasplit:\n  name: train_test\n  groupby: {condition_key}\n  test_size: {float(ccfg['test_size'])}\n  random_state: {int(ccfg['split_random_state'])}\ntraining:\n  n_iters: {int(ccfg['n_iters'])}\n  logs_freq: {int(ccfg['logs_freq'])}\n  eval_freq: {int(ccfg['eval_freq'])}\n"""
    task_file.write_text(task_text, encoding="utf-8")

    print(f"Input shape: {adata.shape}")
    print(f"HVGs: {len(hvg_genes)}")
    print(f"PCA dimensions: {x_pca.shape[1]}")
    print(f"Cumulative variance explained: {pca.explained_variance_ratio_.sum():.6f}")
    print(f"Saved PCA input: {pca_h5ad}")
    print(f"Saved PCA model: {pca_file}")
    print(f"Saved CellOT task config: {task_file}")


if __name__ == "__main__":
    main()
