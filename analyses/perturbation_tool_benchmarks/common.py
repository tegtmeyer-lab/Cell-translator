"""Shared utilities for perturbation benchmark preparation and validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

PERTURBATIONS = ("APOE", "MSI2", "NR1H3", "LRRC4C", "ANKRD12")


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def to_dense_float32(x) -> np.ndarray:
    """Convert dense/sparse matrix-like input to a dense float32 NumPy array."""
    try:
        import scipy.sparse as sp
        if sp.issparse(x):
            x = x.toarray()
    except ImportError:
        pass
    return np.asarray(x, dtype=np.float32)


def read_id_file(path: str | Path) -> list[str]:
    return [line.rstrip("\n") for line in Path(path).read_text().splitlines() if line.strip()]


def write_id_file(ids: Iterable[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(str, ids)) + "\n", encoding="utf-8")


def core_cell_id(value: str) -> str:
    """Reproduce the barcode-normalization rule used for the scGPT matching step."""
    value = str(value)
    match = re.search(r"[ACGT]{16}", value)
    if match:
        return match.group(0)
    match = re.search(r"[ACGT]{10,}", value)
    if match:
        return match.group(0)
    return (
        value.replace("control_neuron_", "")
        .replace("-control_neuron", "")
        .split("-")[0]
    )


def validate_matrix(
    x: np.ndarray,
    *,
    expected_genes: int | None = None,
    expected_cells: int | None = None,
    name: str = "matrix",
) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D; got shape {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if expected_genes is not None and x.shape[1] != expected_genes:
        raise ValueError(f"{name} has {x.shape[1]} genes; expected {expected_genes}")
    if expected_cells is not None and x.shape[0] != expected_cells:
        raise ValueError(f"{name} has {x.shape[0]} cells; expected {expected_cells}")
    return x


def delta_to_absolute_control_mean(
    yhat: np.ndarray,
    control: np.ndarray,
    *,
    clip_nonnegative: bool = True,
) -> np.ndarray:
    """Historical scGPT postprocessing: control mean + predicted delta."""
    yhat = np.asarray(yhat, dtype=np.float32)
    control = np.asarray(control, dtype=np.float32)
    if yhat.ndim != 2 or control.ndim != 2:
        raise ValueError("yhat and control must both be 2D")
    if yhat.shape[1] != control.shape[1]:
        raise ValueError("Gene dimension mismatch between yhat and control")
    out = yhat + control.mean(axis=0, keepdims=True)
    if clip_nonnegative:
        out = np.clip(out, 0.0, None)
    return out.astype(np.float32, copy=False)


def map_to_reference_gene_order(
    x: np.ndarray,
    source_genes: np.ndarray,
    reference_genes: np.ndarray,
) -> np.ndarray:
    """Reorder a complete gene matrix to a reference ordering; fail on missing genes."""
    source_genes = np.asarray(source_genes).astype(str)
    reference_genes = np.asarray(reference_genes).astype(str)
    x = np.asarray(x)
    if x.shape[1] != len(source_genes):
        raise ValueError("Matrix width does not match source gene list")
    if np.array_equal(source_genes, reference_genes):
        return x
    index = {g: i for i, g in enumerate(source_genes)}
    missing = [g for g in reference_genes if g not in index]
    if missing:
        raise ValueError(f"Missing {len(missing)} reference genes; examples: {missing[:5]}")
    order = np.array([index[g] for g in reference_genes], dtype=int)
    return x[:, order]
