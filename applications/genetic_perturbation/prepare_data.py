"""Prepare common control and perturbation expression matrices.

The source notebook used all genes from an already normalized/log1p AnnData
object; no additional normalization or HVG filtering is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scanpy as sc
import yaml


def array_sha256(X: np.ndarray, chunk_rows: int = 256) -> str:
    """Hash an array without materializing an additional full byte copy."""
    X = np.ascontiguousarray(X)
    h = hashlib.sha256()
    for start in range(0, X.shape[0], chunk_rows):
        h.update(memoryview(np.ascontiguousarray(X[start:start + chunk_rows])).cast("B"))
    return h.hexdigest()


def prepare(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    source = Path(cfg["input"]["combined_h5ad"])
    out_dir = Path(cfg["output"]["prepared_data_dir"])
    condition_key = cfg.get("condition_key", "condition")
    source_label = cfg.get("source_label", "ctrl")
    targets = list(cfg["perturbations"])
    threshold = float(cfg.get("threshold", 1e-6))

    adata = sc.read_h5ad(source)
    if condition_key not in adata.obs:
        raise KeyError(f"Condition column '{condition_key}' not found in AnnData.obs")
    found = set(map(str, adata.obs[condition_key].unique()))
    expected = {source_label, *targets}
    missing = expected - found
    if missing:
        raise ValueError(f"Missing required conditions: {sorted(missing)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    gene_names = np.asarray(adata.var_names.astype(str), dtype=str)
    np.save(out_dir / "gene_names.npy", gene_names)

    arrays = {}
    manifest = {
        "source_dataset": "user-supplied AnnData",
        "condition_key": condition_key,
        "normalization": "input already normalized and log1p-transformed; no reprocessing applied",
        "gene_filtering": "all genes retained; no HVG filtering",
        "n_genes": int(adata.n_vars),
        "conditions": {},
    }

    for condition in [source_label, *targets]:
        X = adata[adata.obs[condition_key].astype(str) == condition].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError(f"{condition}: expected finite 2-D expression matrix")
        if np.any(X < 0):
            raise ValueError(f"{condition}: negative values found; expected non-negative log1p data")
        arrays[condition] = X
        filename = "control.npy" if condition == source_label else f"{condition.lower()}.npy"
        np.save(out_dir / filename, X)
        manifest["conditions"][condition] = {
            "n_cells": int(X.shape[0]),
            "n_genes": int(X.shape[1]),
            "zero_fraction": float((X <= threshold).mean()),
            "median_row_sum": float(np.median(X.sum(axis=1))),
            "max": float(X.max()),
            "sha256": array_sha256(X),
            "file": filename,
        }

    # Data-integrity warning: same-shaped perturbations should not normally be
    # byte-identical. This check is especially useful before a manuscript rerun.
    duplicates = []
    for i, a in enumerate(targets):
        for b in targets[i + 1:]:
            if arrays[a].shape == arrays[b].shape and np.array_equal(arrays[a], arrays[b]):
                duplicates.append([a, b])
    manifest["exact_duplicate_condition_pairs"] = duplicates
    if duplicates:
        print(f"WARNING: exact duplicate perturbation matrices detected: {duplicates}")

    (out_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Preparation YAML file")
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
