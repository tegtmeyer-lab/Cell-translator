"""Generate the consolidated held-out result suite for one perturbation.

This module evaluates held-out Cell Translator predictions using the
application's predefined gene-expression and latent-space metrics.
Numerical definitions and embedding settings are preserved across
perturbations, while plot labels are generated from the selected
perturbation rather than hard-coded for individual genes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from cell_translator.inference.perturbation import load_perturbation_latents
from config_utils import load_config
from evaluation import (
    historical_lfc_log1p_of_mean,
    per_gene_ks,
    per_gene_wasserstein,
)

SEED = 42


def _save_or_show(fig, path: Path, show: bool):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_latent_embeddings(Z_source, Z_real, Z_pred, gene, out_path, show=False):
    """PCA/t-SNE/UMAP settings match the repeated notebook result suites."""
    try:
        import umap
    except ImportError as exc:
        raise ImportError("UMAP plotting requires the 'umap-learn' package.") from exc

    n_source = min(2000, len(Z_source))
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(Z_source), n_source, replace=False)
    Z_all = np.concatenate([Z_source[idx], Z_real, Z_pred]).astype(np.float32)
    labels = np.asarray(
        ["control"] * n_source + [f"real {gene}"] * len(Z_real) + [f"predicted {gene}"] * len(Z_pred)
    )

    pca_2d = PCA(n_components=2, random_state=SEED).fit_transform(Z_all)
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=SEED, n_jobs=-1).fit_transform(Z_all)
    umap_2d = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=15, min_dist=0.3).fit_transform(Z_all)

    colors = {"control": "#FCD34D", f"real {gene}": "#FCA5A5", f"predicted {gene}": "#A8DADC"}
    sizes = {"control": 5, f"real {gene}": 12, f"predicted {gene}": 10}
    alphas = {"control": 0.3, f"real {gene}": 0.8, f"predicted {gene}": 0.7}

    fig, axes = plt.subplots(1, 3, figsize=(10, 5), dpi=300)
    for ax, title, emb in zip(axes, ["PCA", "t-SNE", "UMAP"], [pca_2d, tsne_2d, umap_2d]):
        for label, color in colors.items():
            mask = labels == label
            ax.scatter(emb[mask, 0], emb[mask, 1], c=color, s=sizes[label], alpha=alphas[label], rasterized=True)
        ax.set_title(title)
        ax.set_xlabel(f"{title} 1")
        ax.set_ylabel(f"{title} 2")
        ax.grid(alpha=0.2)
    axes[-1].legend(
        handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=l) for l, c in colors.items()],
        framealpha=0.9,
        fontsize=9,
    )
    fig.suptitle(f"Latent Space: control | real {gene} | predicted {gene}", fontsize=13, fontweight="bold")
    _save_or_show(fig, out_path, show)


def plot_correlations(X_real, X_pred, gene, threshold, out_path, show=False):
    mean_real, mean_pred = X_real.mean(0), X_pred.mean(0)
    det_real, det_pred = (X_real > threshold).mean(0), (X_pred > threshold).mean(0)
    r_mean, p_mean = pearsonr(mean_real, mean_pred)
    r_det, p_det = pearsonr(det_real, det_pred)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    for ax, xv, yv, r, p, title in [
        (axes[0], mean_real, mean_pred, r_mean, p_mean, "Gene Mean Expression"),
        (axes[1], det_real, det_pred, r_det, p_det, "Detection Rate"),
    ]:
        mn, mx = min(xv.min(), yv.min()), max(xv.max(), yv.max())
        ax.scatter(xv, yv, s=5, alpha=0.3, rasterized=True)
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.2)
        p_str = f"{p:.2e}" if p > 0 else "< 1e-300"
        ax.set_title(title)
        ax.annotate(f"Pearson r = {r:.4f}\np = {p_str}", xy=(0.05, 0.88), xycoords="axes fraction", fontsize=10)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Real vs Predicted {gene} — Cell Translator", fontsize=13, fontweight="bold")
    _save_or_show(fig, out_path, show)


def plot_ks_and_wasserstein(ks_df, wd_df, gene, out_dir, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    values = np.sort(ks_df.ks_stat.to_numpy())
    axes[0].plot(values, np.arange(1, len(values) + 1) / len(values), lw=2)
    axes[0].axvline(np.median(values), color="red", ls="--", lw=1.5, label=f"Median={np.median(values):.3f}")
    axes[0].axvline(0.1, color="gray", ls=":", lw=1, label="KS=0.1 threshold")
    axes[0].set_xlabel("KS Statistic (per gene)"); axes[0].set_ylabel("CDF"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].hist(ks_df.ks_stat.to_numpy(), bins=80, edgecolor="white", linewidth=0.3, alpha=0.85)
    axes[1].axvline(np.median(values), color="red", ls="--", lw=1.5)
    axes[1].set_xlabel("KS Statistic"); axes[1].set_ylabel("Gene count"); axes[1].grid(alpha=0.3)
    fig.suptitle(f"KS Test: Real {gene} vs Predicted {gene}", fontsize=13, fontweight="bold")
    _save_or_show(fig, out_dir / "per_gene_ks.png", show)

    vals = wd_df.wasserstein.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    axes[0].plot(np.arange(len(vals)), np.sort(vals), lw=1.5)
    axes[0].axhline(np.median(vals), color="red", ls="--", lw=1.5, label=f"Median={np.median(vals):.3f}")
    axes[0].set_xlabel("Genes (sorted)"); axes[0].set_ylabel("Wasserstein distance"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].hist(vals, bins=80, edgecolor="white", linewidth=0.3, alpha=0.85)
    axes[1].axvline(np.median(vals), color="red", ls="--", lw=1.5)
    axes[1].set_xlabel("Wasserstein distance"); axes[1].set_ylabel("Gene count"); axes[1].grid(alpha=0.3)
    fig.suptitle(f"Wasserstein Distance: Real {gene} vs Predicted {gene}", fontsize=13, fontweight="bold")
    _save_or_show(fig, out_dir / "per_gene_wasserstein.png", show)


def plot_historical_lfc(X_source, X_real, X_pred, genes, gene, out_path, top_n=200, show=False):
    real = historical_lfc_log1p_of_mean(X_source, X_real)
    pred = historical_lfc_log1p_of_mean(X_source, X_pred)
    r, p = pearsonr(real, pred)
    rs = spearmanr(real, pred).correlation
    top_idx = np.argsort(np.abs(real))[-top_n:]
    gene_arr = np.asarray(genes).astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    axes[0].scatter(real, pred, s=4, alpha=0.25, rasterized=True)
    axes[0].scatter(real[top_idx], pred[top_idx], s=18, alpha=0.7, label=f"Top-{top_n} DEGs")
    mn, mx = min(real.min(), pred.min()), max(real.max(), pred.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1.2)
    axes[0].annotate(f"Pearson r={r:.4f}\nSpearman r={rs:.4f}", xy=(0.04, 0.86), xycoords="axes fraction")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.25)
    axes[1].scatter(real[top_idx], pred[top_idx], s=20, alpha=0.7)
    mn2, mx2 = min(real[top_idx].min(), pred[top_idx].min()), max(real[top_idx].max(), pred[top_idx].max())
    axes[1].plot([mn2, mx2], [mn2, mx2], "r--", lw=1.2)
    dev = np.abs(pred[top_idx] - real[top_idx])
    for i in np.argsort(dev)[-15:]:
        axes[1].annotate(gene_arr[top_idx[i]], (real[top_idx[i]], pred[top_idx[i]]), fontsize=6, xytext=(4, 4), textcoords="offset points")
    axes[1].set_title(f"Top-{top_n} DEGs")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"Historical LFC Agreement: Real vs Predicted {gene}", fontsize=13, fontweight="bold")
    _save_or_show(fig, out_path, show)


def plot_marker_violins(X_source, X_real, X_pred, genes, markers, gene, out_dir, max_cells=1500, show=False):
    g2i = {g: i for i, g in enumerate(np.asarray(genes).astype(str))}
    rng = np.random.default_rng(SEED)
    def sub(X):
        return X[rng.choice(len(X), min(max_cells, len(X)), replace=False)]
    Xs, Xr, Xp = sub(X_source), sub(X_real), sub(X_pred)
    marker_dir = out_dir / "marker_violins"; marker_dir.mkdir(parents=True, exist_ok=True)
    for marker in [m for m in markers if m in g2i]:
        j = g2i[marker]
        data = [Xs[:, j], Xr[:, j], Xp[:, j]]
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        parts = ax.violinplot(data, positions=[1, 2, 3], showmeans=True, showextrema=True, widths=0.7)
        for key in ["cmeans", "cmaxes", "cmins", "cbars"]:
            if key in parts:
                parts[key].set_color("black"); parts[key].set_linewidth(1.2)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["control", f"real {gene}", f"predicted {gene}"])
        ax.set_ylabel("Expression (log1p)"); ax.set_title(marker, fontweight="bold"); ax.grid(alpha=0.2, axis="y")
        _save_or_show(fig, marker_dir / f"violin_{marker}.png", show)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", required=True, choices=["APOE", "MSI2", "NR1H3", "LRRC4C", "ANKRD12"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg["gene"] != args.gene:
        raise ValueError("Gene/config mismatch")
    out_dir = Path(cfg["outputs"]["directory"]) / "result_suite"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.predictions) if args.predictions else Path(cfg["outputs"]["directory"]) / "translated_validation_outputs.npz"
    pred = np.load(pred_path, allow_pickle=False)
    X_pred = pred["X_pred_hard"].astype(np.float32)
    src_idx = pred["source_validation_indices"].astype(int)
    tgt_idx = pred["target_validation_indices"].astype(int)
    X_source_all = np.load(cfg["data"]["control_expression"]).astype(np.float32)
    X_real_all = np.load(cfg["data"]["target_expression"]).astype(np.float32)
    genes = np.load(cfg["data"]["gene_names"], allow_pickle=False).astype(str)
    X_source, X_real = X_source_all[src_idx], X_real_all[tgt_idx]
    lat = load_perturbation_latents(cfg["archived_artifacts"]["ae_outputs"], args.gene)
    Z_source = lat["val_src"] * lat["std_ctrl"] + lat["mean_ctrl"]
    Z_real = lat["val_tgt"] * lat["std_target"] + lat["mean_target"]
    Z_pred = pred["translated_target_latent_standardized"] * lat["std_target"] + lat["mean_target"]

    plot_latent_embeddings(Z_source, Z_real, Z_pred, args.gene, out_dir / "latent_embeddings.png", args.show_plots)
    plot_correlations(X_real, X_pred, args.gene, float(cfg["evaluation"]["threshold"]), out_dir / "correlations.png", args.show_plots)
    ks_df = per_gene_ks(X_real, X_pred, genes)
    wd_df = per_gene_wasserstein(X_real, X_pred, genes)
    plot_ks_and_wasserstein(ks_df, wd_df, args.gene, out_dir, args.show_plots)
    plot_historical_lfc(X_source, X_real, X_pred, genes, args.gene, out_dir / "historical_lfc.png", 200, args.show_plots)
    plot_marker_violins(X_source, X_real, X_pred, genes, cfg.get("markers_used_in_single_perturbation_suite", []), args.gene, out_dir, 1500, args.show_plots)


if __name__ == "__main__":
    main()
