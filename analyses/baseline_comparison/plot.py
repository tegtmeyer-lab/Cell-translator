#!/usr/bin/env python
"""Generate compact baseline-comparison figures from saved benchmark metrics."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHOD_ORDER = ["Identity", "MeanShift", "NearestNeighbor", "LinearLatentShift", "CellTranslator"]
KEY_METRICS = [
    ("effect_pearson", "Effect-size Pearson r", True),
    ("sign_agreement_top500", "Top-500 sign agreement", True),
    ("median_gene_ks", "Median gene-wise KS", False),
    ("pred_vs_real_auc", "Predicted-vs-real AUC", False),
]


def make_summary_figure(df, outdir):
    summary = df.groupby("method").agg(
        {m[0]: ["mean", "std"] for m in KEY_METRICS}
    )
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
    methods = [m for m in METHOD_ORDER if m in summary.index]
    y = np.arange(len(methods))
    for ax, (metric, label, _) in zip(axes, KEY_METRICS):
        means = np.array([summary.loc[m, (metric, "mean")] for m in methods], float)
        stds = np.array([summary.loc[m, (metric, "std")] for m in methods], float)
        stds[~np.isfinite(stds)] = 0
        for i, (method, mean, sd) in enumerate(zip(methods, means, stds)):
            if np.isfinite(mean):
                ax.barh(i, mean, xerr=sd, height=0.55)
            else:
                ax.text(0.02, i, "NA", va="center", transform=ax.get_yaxis_transform())
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.25)
        if metric == "pred_vs_real_auc":
            ax.axvline(0.5, ls="--", lw=1)
        ax.set_yticks(y, methods)
    fig.suptitle("Cell Translator versus simple perturbation baselines")
    fig.tight_layout()
    fig.savefig(outdir / "baseline_summary_metrics.pdf", bbox_inches="tight")
    fig.savefig(outdir / "baseline_summary_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_heatmaps(df, outdir):
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    perturbations = sorted(df["perturbation"].unique())
    for metric, label, _ in KEY_METRICS:
        pivot = df.pivot(index="perturbation", columns="method", values=metric).reindex(
            index=perturbations, columns=methods
        )
        fig, ax = plt.subplots(figsize=(1.5 * len(methods) + 2, 0.7 * len(perturbations) + 2))
        arr = np.ma.masked_invalid(pivot.to_numpy(float))
        im = ax.imshow(arr, aspect="auto")
        im.cmap.set_bad("0.9")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = pivot.iloc[i, j]
                ax.text(j, i, "NA" if not np.isfinite(val) else f"{val:.2f}", ha="center", va="center")
        ax.set_xticks(range(len(methods)), methods, rotation=30, ha="right")
        ax.set_yticks(range(len(perturbations)), perturbations)
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.tight_layout()
        fig.savefig(outdir / f"baseline_heatmap_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)
    make_summary_figure(df, outdir)
    make_heatmaps(df, outdir)
    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
