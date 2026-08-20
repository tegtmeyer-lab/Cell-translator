"""Train scGen and predict all five perturbations for the common test-control cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import load_yaml, read_id_file, to_dense_float32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    scfg = cfg["scgen"]

    import anndata as ad
    import scipy.sparse as sp
    import scgen
    import scvi

    adata = ad.read_h5ad(cfg["dataset"]["h5ad"])
    test_ids = read_id_file(cfg["common_test_control_ids"])
    adata_test = adata[test_ids].copy()

    if adata_test.n_obs != int(cfg["dataset"]["expected_test_control_cells"]):
        raise ValueError(f"Expected {cfg['dataset']['expected_test_control_cells']} test controls; got {adata_test.n_obs}")

    adata.X = to_dense_float32(adata.X)
    adata_test.X = to_dense_float32(adata_test.X)

    condition_key = cfg["dataset"].get("condition_key", "condition")
    control_label = cfg["dataset"].get("control_label", "ctrl")
    scgen.SCGEN.setup_anndata(adata, batch_key=condition_key)

    seed = scfg.get("seed")
    if seed is not None:
        scvi.settings.seed = int(seed)

    model = scgen.SCGEN(adata)
    model.train(max_epochs=int(scfg["max_epochs"]))

    outroot = Path(scfg["output_root"])
    for gene in cfg["perturbations"]:
        pred, _delta = model.predict(
            ctrl_key=control_label,
            stim_key=gene,
            adata_to_predict=adata_test,
        )
        outdir = outroot / gene
        outdir.mkdir(parents=True, exist_ok=True)
        pred.write_h5ad(outdir / f"scgen_pred_{gene}_testctrl.h5ad")
        x = to_dense_float32(pred.X)
        np.save(outdir / f"scgen_pred_{gene}_testctrl.npy", x)
        print(f"[{gene}] saved shape={x.shape} range=({x.min():.6f}, {x.max():.6f})")

    print(f"scgen version: {getattr(scgen, '__version__', 'unknown')}")
    print(f"scvi-tools version: {getattr(scvi, '__version__', 'unknown')}")


if __name__ == "__main__":
    main()
