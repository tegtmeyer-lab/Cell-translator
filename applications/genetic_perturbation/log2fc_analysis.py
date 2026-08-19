"""Dedicated linear-space log2 fold-change analysis for perturbation prediction.

This consolidates the final repeated per-perturbation log2FC analysis. Inputs
are log1p expression matrices. Expression is converted back to linear space,
averaged per gene, and compared using a log2 ratio with a pseudocount.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from config_utils import load_config


def safe_pearsonr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(pearsonr(a, b)[0])


def compute_linear_log2fc(
    X_control_log1p,
    X_real_log1p,
    X_pred_log1p,
    gene_names,
    *,
    pseudocount=1e-3,
    detection_threshold=0.0,
):
    """Reproduce the final notebook linear-space log2FC calculation."""
    Xc = np.asarray(X_control_log1p, dtype=np.float32)
    Xr = np.asarray(X_real_log1p, dtype=np.float32)
    Xp = np.asarray(X_pred_log1p, dtype=np.float32)
    genes = np.asarray(gene_names).astype(str)
    if any(x.ndim != 2 for x in (Xc, Xr, Xp)):
        raise ValueError("Expression inputs must be 2-D cells × genes matrices.")
    if not (Xc.shape[1] == Xr.shape[1] == Xp.shape[1] == len(genes)):
        raise ValueError("Gene dimensions/order are inconsistent across inputs.")

    det_c = (Xc > detection_threshold).mean(axis=0)
    det_r = (Xr > detection_threshold).mean(axis=0)
    det_p = (Xp > detection_threshold).mean(axis=0)

    mean_c = np.expm1(Xc).mean(axis=0)
    mean_r = np.expm1(Xr).mean(axis=0)
    mean_p = np.expm1(Xp).mean(axis=0)
    pc = float(pseudocount)
    true_fc = np.log2((mean_r + pc) / (mean_c + pc))
    pred_fc = np.log2((mean_p + pc) / (mean_c + pc))

    return pd.DataFrame({
        "gene": genes,
        "control_mean_linear": mean_c,
        "real_target_mean_linear": mean_r,
        "predicted_target_mean_linear": mean_p,
        "true_log2fc": true_fc,
        "pred_log2fc": pred_fc,
        "log2fc_error": np.abs(pred_fc - true_fc),
        "control_detect": det_c,
        "real_target_detect": det_r,
        "predicted_target_detect": det_p,
    })


def plot_log2fc(df, target_name, output_path, *, pseudocount=1e-3, detect_min=0.01, markers=None):
    """Four-panel visualization matching the final repeated notebook analysis."""
    pc = float(pseudocount)
    err = df["log2fc_error"].to_numpy()
    vmax = float(np.quantile(err, 0.95))
    x_ctrl = np.log2(df["control_mean_linear"].to_numpy() + pc)
    x_real = np.log2(df["real_target_mean_linear"].to_numpy() + pc)
    x_pred = np.log2(df["predicted_target_mean_linear"].to_numpy() + pc)
    m_real = (df["control_detect"].to_numpy() > detect_min) & (df["real_target_detect"].to_numpy() > detect_min)
    m_pred = (df["control_detect"].to_numpy() > detect_min) & (df["predicted_target_detect"].to_numpy() > detect_min)

    r_cr = safe_pearsonr(x_ctrl[m_real], x_real[m_real])
    r_cp = safe_pearsonr(x_ctrl[m_pred], x_pred[m_pred])
    r_fc = safe_pearsonr(df["true_log2fc"], df["pred_log2fc"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=300)
    panels = [
        (axes[0,0], x_ctrl[m_real], x_real[m_real], err[m_real], f"log2 mean(Control) vs log2 mean(Real {target_name})", r_cr),
        (axes[0,1], x_ctrl[m_pred], x_pred[m_pred], err[m_pred], f"log2 mean(Control) vs log2 mean(Predicted {target_name})", r_cp),
    ]
    for ax, x, y, c, title, r in panels:
        sc = ax.scatter(x, y, s=24, alpha=0.45, c=c, vmin=0, vmax=vmax)
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.4, alpha=0.6)
        ax.text(0.04, 0.96, f"r = {r:.3f}", transform=ax.transAxes, va="top", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, label="|Pred log2FC − True log2FC|")

    ax = axes[1,0]
    true_fc, pred_fc = df["true_log2fc"].to_numpy(), df["pred_log2fc"].to_numpy()
    sc = ax.scatter(true_fc, pred_fc, s=24, alpha=0.45, c=err, vmin=0, vmax=vmax)
    lo, hi = float(min(true_fc.min(), pred_fc.min())), float(max(true_fc.max(), pred_fc.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.4, alpha=0.6)
    ax.text(0.04, 0.96, f"r = {r_fc:.3f}", transform=ax.transAxes, va="top", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax.set_title(f"Control → {target_name}: True vs Predicted log2FC", fontsize=13, fontweight="bold")
    ax.set_xlabel("True log2FC")
    ax.set_ylabel("Predicted log2FC")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, label="|Pred − True| (log2FC)")

    if markers:
        for gene in markers:
            sub = df[df["gene"] == gene]
            if sub.empty:
                continue
            row = sub.iloc[0]
            ax.scatter(row["true_log2fc"], row["pred_log2fc"], s=260, marker="*", edgecolor="black", lw=1.0, zorder=6)
            ax.annotate(gene, (row["true_log2fc"], row["pred_log2fc"]), xytext=(10,10), textcoords="offset points", fontsize=9)

    ax = axes[1,1]
    ax.scatter(np.abs(true_fc), err, s=18, alpha=0.45)
    ax.set_title("log2FC error vs |true log2FC|", fontsize=13, fontweight="bold")
    ax.set_xlabel("|True log2FC|")
    ax.set_ylabel("|Pred log2FC − True log2FC|")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {"log2fc_pearson_r": r_fc, "kept_real_fraction": float(m_real.mean()), "kept_pred_fraction": float(m_pred.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", required=True, choices=["APOE", "MSI2", "NR1H3", "LRRC4C", "ANKRD12"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", default=None, help="translated_validation_outputs.npz; defaults to the configured output directory")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["gene"] != args.gene:
        raise ValueError("Gene/config mismatch")

    X_ctrl = np.load(cfg["data"]["control_expression"]).astype(np.float32)
    X_real = np.load(cfg["data"]["target_expression"]).astype(np.float32)
    genes = np.load(cfg["data"]["gene_names"], allow_pickle=False)
    pred_path = args.predictions or str(Path(cfg["outputs"]["directory"]) / "translated_validation_outputs.npz")
    pred = np.load(pred_path, allow_pickle=False)
    src_idx = pred["source_validation_indices"].astype(int)
    tgt_idx = pred["target_validation_indices"].astype(int)
    X_pred = pred["X_pred_hard"].astype(np.float32)

    ev = cfg["evaluation"]
    df = compute_linear_log2fc(
        X_ctrl[src_idx], X_real[tgt_idx], X_pred, genes,
        pseudocount=float(ev["dedicated_log2fc_pseudocount"]), detection_threshold=0.0,
    )
    out = Path(cfg["outputs"]["directory"]) / "log2fc"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "linear_space_log2fc.csv", index=False)
    summary = plot_log2fc(
        df, args.gene, out / "linear_space_log2fc.png",
        pseudocount=float(ev["dedicated_log2fc_pseudocount"]),
        detect_min=float(ev["dedicated_log2fc_min_detection"]),
        markers=cfg.get("markers_used_in_combined_comparison", []),
    )
    print(summary)


if __name__ == "__main__":
    main()
