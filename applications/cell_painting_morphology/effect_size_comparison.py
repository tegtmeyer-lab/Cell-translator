"""Effect-size recovery and control/deletion comparison for morphology translation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from config_utils import load_config


def cohens_d_per_feature(source, target, eps=1e-8):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mean_src = source.mean(0)
    mean_tgt = target.mean(0)
    var_src = source.var(0, ddof=1)
    var_tgt = target.var(0, ddof=1)
    n1, n2 = len(source), len(target)
    pooled = np.sqrt((((n1 - 1) * var_src) + ((n2 - 1) * var_tgt)) / max(n1 + n2 - 2, 1))
    pooled = np.where(pooled < eps, eps, pooled)
    return (mean_tgt - mean_src) / pooled


def _original_matrix(prepared_dir, state, features):
    X = np.load(Path(prepared_dir) / f"{state}_X.npy").astype(np.float32)
    if X.shape[1] != len(features):
        raise ValueError(f"Feature count mismatch for {state}: {X.shape[1]} vs {len(features)}")
    # Historical downstream effect-size analysis converted missing/Inf to zero.
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _translated(output_dir, source, target):
    return np.load(Path(output_dir) / f"{source}_to_{target}" / "translated_features.npy").astype(np.float32)


def compare(control_cfg, deletion_cfg, output_path):
    control_features = pd.read_csv(Path(control_cfg["paths"]["prepared_dir"]) / "common_feature_columns.csv")["feature"].astype(str).tolist()
    deletion_features = pd.read_csv(Path(deletion_cfg["paths"]["prepared_dir"]) / "common_feature_columns.csv")["feature"].astype(str).tolist()
    shared = sorted(set(control_features).intersection(deletion_features))
    if control_cfg["effect_size"].get("remove_neighbor_features", True):
        shared = [f for f in shared if "neighbor" not in f.lower()]

    ctrl_idx = [control_features.index(f) for f in shared]
    del_idx = [deletion_features.index(f) for f in shared]
    rows = []

    deletion_transition_map = {
        t["label"]: t for t in deletion_cfg.get("transitions", []) if t.get("enabled", True)
    }

    for trans in control_cfg["transitions"]:
        if not trans.get("enabled", True) or trans["label"] not in deletion_transition_map:
            continue
        dtrans = deletion_transition_map[trans["label"]]

        src_c = _original_matrix(control_cfg["paths"]["prepared_dir"], trans["source"], control_features)[:, ctrl_idx]
        tgt_c = _original_matrix(control_cfg["paths"]["prepared_dir"], trans["target"], control_features)[:, ctrl_idx]
        tr_c = _translated(control_cfg["paths"]["output_dir"], trans["source"], trans["target"])[:, ctrl_idx]

        src_d = _original_matrix(deletion_cfg["paths"]["prepared_dir"], dtrans["source"], deletion_features)[:, del_idx]
        tgt_d = _original_matrix(deletion_cfg["paths"]["prepared_dir"], dtrans["target"], deletion_features)[:, del_idx]
        tr_d = _translated(deletion_cfg["paths"]["output_dir"], dtrans["source"], dtrans["target"])[:, del_idx]

        c_meas = cohens_d_per_feature(src_c, tgt_c)
        c_pred = cohens_d_per_feature(src_c, tr_c)
        d_meas = cohens_d_per_feature(src_d, tgt_d)
        d_pred = cohens_d_per_feature(src_d, tr_d)

        comparisons = {
            "control_measured_vs_translated": (c_meas, c_pred),
            "deletion_measured_vs_translated": (d_meas, d_pred),
            "control_vs_deletion_measured": (c_meas, d_meas),
            "control_vs_deletion_translated": (c_pred, d_pred),
        }
        for name, (x, y) in comparisons.items():
            mask = np.isfinite(x) & np.isfinite(y)
            rows.append({
                "transition": trans["label"],
                "comparison": name,
                "pearson_r": float(pearsonr(x[mask], y[mask])[0]),
                "spearman_r": float(spearmanr(x[mask], y[mask])[0]),
                "n_features": int(mask.sum()),
            })

    result = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-config", required=True)
    parser.add_argument("--deletion-config", required=True)
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV. Defaults to <deletion output_dir>/effect_size_comparison.csv.",
    )
    args = parser.parse_args()
    control_cfg = load_config(args.control_config)
    deletion_cfg = load_config(args.deletion_config)
    output = args.output or str(
        Path(deletion_cfg["paths"]["output_dir"]) / "effect_size_comparison.csv"
    )
    result = compare(control_cfg, deletion_cfg, output)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
