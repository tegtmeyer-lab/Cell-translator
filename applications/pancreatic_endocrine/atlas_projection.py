"""Project translated Endocrine cells onto the source pancreatic atlas UMAP."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from config_utils import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    pcfg = cfg["atlas_projection"]

    atlas = ad.read_h5ad(cfg["preparation"]["input_h5ad"])
    cluster_col = cfg["preparation"]["cluster_column"]
    day = pd.to_numeric(
        atlas.obs[cfg["preparation"]["day_column"]].astype(str).str.extract(r"(\d+\.?\d*)")[0],
        errors="coerce",
    )
    source_mask = (
        (atlas.obs[cluster_col].astype(str) == cfg["preparation"]["source_cluster_label"])
        & (day <= float(cfg["preparation"]["source_day_max"]))
    )
    target_mask = (
        (atlas.obs[cluster_col].astype(str) == cfg["preparation"]["target_cluster_label"])
        & (day >= float(cfg["preparation"]["target_day_min"]))
    )

    prepared = Path(cfg["paths"]["prepared_dir"])
    out = Path(cfg["paths"]["output_dir"])
    genes = np.load(prepared / "hvg_genes.npy", allow_pickle=False).astype(str)
    slug = f"{cfg['source_state'].lower()}_to_{cfg['target_state'].lower()}".replace(" ", "_")
    translated = np.load(out / slug / "translated_source_outputs.npz", allow_pickle=False)["X_translated"].astype(np.float32)

    ref = atlas[:, genes].X
    if sparse.issparse(ref):
        ref = ref.toarray()
    ref = np.asarray(ref, np.float32)
    coords = np.asarray(atlas.obsm[pcfg["umap_key"]], np.float32)

    nn = NearestNeighbors(
        n_neighbors=int(pcfg["n_neighbors"]),
        metric=pcfg["metric"],
        n_jobs=int(pcfg["n_jobs"]),
    ).fit(ref)
    _, indices = nn.kneighbors(translated)
    projected = coords[indices].mean(axis=1)

    target_idx = np.where(target_mask.to_numpy())[0]
    neighbor_labels = np.asarray(atlas.obs[cluster_col].astype(str))[indices]
    counts = Counter(neighbor_labels.ravel().tolist())
    summary = {
        "translated_cells": int(translated.shape[0]),
        "measured_target_centroid": coords[target_idx].mean(0).tolist(),
        "translated_centroid": projected.mean(0).tolist(),
        "neighbor_label_counts": dict(counts),
    }
    with (out / "atlas_projection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Compact public figure: full atlas with source/target/translated overlay.
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    other = ~(source_mask | target_mask).to_numpy()
    ax.scatter(coords[other,0], coords[other,1], s=6, alpha=0.18, label="Other measured cells")
    ax.scatter(coords[source_mask,0], coords[source_mask,1], s=12, alpha=0.75, label="Measured NGN3-high EP")
    ax.scatter(coords[target_mask,0], coords[target_mask,1], s=12, alpha=0.75, label="Measured Endocrine")
    ax.scatter(projected[:,0], projected[:,1], s=18, alpha=0.85, label="Translated Endocrine")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title("NGN3 to Endocrine translation on pancreatic atlas")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "atlas_translated_overlay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
