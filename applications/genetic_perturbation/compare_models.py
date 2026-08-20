"""Consolidated final model-comparison analysis across five perturbations.

This replaces the repeated exploratory comparison cells with one implementation of the final per-gene KS/Wasserstein, detection-rate,
and top-DE effect-recovery metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation import final_comparison_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    threshold = float(cfg.get("threshold", 1e-6))
    top_de = int(cfg.get("top_de_genes", 5000))
    ks_bad = float(cfg.get("ks_bad_threshold", 0.2))

    rows = []
    per_gene_rows = []
    for perturbation, item in cfg["perturbations"].items():
        X_ctrl = np.load(item["control"]).astype(np.float32)
        X_real = np.load(item["real"]).astype(np.float32)
        for model, pred_path in item["predictions"].items():
            X_pred = np.load(pred_path).astype(np.float32)
            if X_pred.shape[1] != X_real.shape[1] or X_ctrl.shape[1] != X_real.shape[1]:
                raise ValueError(f"Gene dimension mismatch for {perturbation}/{model}")
            # The final notebook comparison used the measured control matrix as
            # the reference control for both real and predicted effects.
            m = final_comparison_metrics(
                X_ctrl, X_ctrl, X_real, X_pred,
                top_de=top_de, threshold=threshold, ks_bad=ks_bad,
            )
            rows.append({"perturbation": perturbation, "model": model, **m})

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "model_comparison_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
