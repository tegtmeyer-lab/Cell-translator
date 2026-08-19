"""Validate or create preprocessed acute-leukemia Cell Translator inputs.

The supplied model notebooks begin from preprocessed log1p-normalized matrices
with 5,000 genes. They do not contain the upstream raw-count QC/HVG-selection
pipeline; this module therefore validates those model inputs rather than
inventing an unreported preprocessing procedure.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def validate_matrix(x: np.ndarray, name: str, expected_genes: int = 5000) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"{name}: expected 2D matrix, got {x.shape}.")
    if x.shape[1] != expected_genes:
        raise ValueError(f"{name}: expected {expected_genes} genes, got {x.shape[1]}.")
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: matrix contains NaN/Inf values.")
    if (x < 0).any():
        raise ValueError(f"{name}: expected non-negative log1p expression values.")
    return x


def csv_pair_to_numpy(source_csv: str, target_csv: str, output_dir: str):
    src = pd.read_csv(source_csv, index_col=0)
    tgt = pd.read_csv(target_csv, index_col=0)
    if list(src.columns) != list(tgt.columns):
        raise ValueError("Source and target CSV gene columns are not identical and ordered the same way.")
    x_src = validate_matrix(src.to_numpy(np.float32), "source")
    x_tgt = validate_matrix(tgt.to_numpy(np.float32), "target")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "source.npy", x_src)
    np.save(out / "target.npy", x_tgt)
    np.save(out / "gene_names.npy", np.asarray(src.columns, dtype=str))
    print(f"Saved {x_src.shape} source and {x_tgt.shape} target matrices to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-npy")
    ap.add_argument("--target-npy")
    ap.add_argument("--source-csv")
    ap.add_argument("--target-csv")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    if args.source_csv and args.target_csv:
        csv_pair_to_numpy(args.source_csv, args.target_csv, args.output_dir)
        return
    if not (args.source_npy and args.target_npy):
        raise SystemExit("Provide either --source-csv/--target-csv or --source-npy/--target-npy.")
    src = validate_matrix(np.load(args.source_npy), "source")
    tgt = validate_matrix(np.load(args.target_npy), "target")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "source.npy", src); np.save(out / "target.npy", tgt)
    print(f"Validated source {src.shape} and target {tgt.shape}.")


if __name__ == "__main__":
    main()
