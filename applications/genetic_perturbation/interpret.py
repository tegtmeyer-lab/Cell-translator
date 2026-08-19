"""Integrated-gradients driver-gene analysis for genetic perturbations.

The repeated gene-specific notebook implementations are consolidated here. The
IG objective and DriverScore formula are preserved exactly; the code names the
historical DriverScore fold-change calculation explicitly because it differs
from the dedicated linear-space log2FC analysis used elsewhere in the notebook.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_translator.inference.perturbation import load_perturbation_ae_checkpoint, load_perturbation_latents
from config_utils import load_config


class TorchForwardGenerator(nn.Module):
    """PyTorch mirror of the Keras forward generator including affine head."""
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 1024); self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512); self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 512); self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, d)
        self.head = nn.Linear(d, d)

    def forward(self, z):
        z = self.ln1(F.relu(self.fc1(z)))
        z = self.ln2(F.relu(self.fc2(z)))
        z = self.ln3(F.relu(self.fc3(z)))
        return self.head(self.fc4(z))


def _collect_dense_ln_in_order(model):
    import tensorflow as tf
    out = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            out.extend(_collect_dense_ln_in_order(layer))
        elif isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.LayerNormalization)):
            out.append(layer)
    return out


def load_keras_generator_into_torch(keras_model, torch_model):
    """Transfer the five Dense and three LayerNorm layers in notebook order."""
    import tensorflow as tf
    layers = _collect_dense_ln_in_order(keras_model)
    dense = [x for x in layers if isinstance(x, tf.keras.layers.Dense)]
    norms = [x for x in layers if isinstance(x, tf.keras.layers.LayerNormalization)]
    if len(dense) < 5 or len(norms) < 3:
        raise ValueError(f"Unexpected generator layer counts: Dense={len(dense)}, LN={len(norms)}")

    def set_linear(dst, src):
        W, b = src.get_weights()
        dst.weight.data = torch.tensor(W.T, dtype=torch.float32)
        dst.bias.data = torch.tensor(b, dtype=torch.float32)

    def set_ln(dst, src):
        gamma, beta = src.get_weights()
        dst.weight.data = torch.tensor(gamma, dtype=torch.float32)
        dst.bias.data = torch.tensor(beta, dtype=torch.float32)

    set_linear(torch_model.fc1, dense[0]); set_ln(torch_model.ln1, norms[0])
    set_linear(torch_model.fc2, dense[1]); set_ln(torch_model.ln2, norms[1])
    set_linear(torch_model.fc3, dense[2]); set_ln(torch_model.ln3, norms[2])
    set_linear(torch_model.fc4, dense[3]); set_linear(torch_model.head, dense[4])
    return torch_model


@torch.no_grad()
def make_baseline(X, mode="mean"):
    if mode == "zero":
        return torch.zeros_like(X)
    if mode == "mean":
        return X.mean(dim=0, keepdim=True).repeat(X.shape[0], 1)
    raise ValueError("baseline_mode must be 'mean' or 'zero'")


def integrated_gradients_latent_shift(
    encoder,
    generator,
    X_source,
    gene_names,
    mean_source,
    std_source,
    *,
    steps=64,
    max_samples=1024,
    baseline_mode="mean",
    batch_size=16,
    seed=42,
    latent_std_epsilon=1e-6,
    device=None,
):
    """Per-gene IG for the notebook objective 0.5*||G(z_std)-z_std||^2."""
    device = device or torch.device("cpu")
    encoder.eval(); generator.eval()
    X = np.asarray(X_source, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_samples:
        X = X[rng.choice(X.shape[0], size=max_samples, replace=False)]

    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    B = make_baseline(X_t, mode=baseline_mode)
    mean = torch.as_tensor(mean_source, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_source, dtype=torch.float32, device=device)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)
    delta = 1.0 / steps
    attr_sum_abs = torch.zeros(X_t.shape[1], device=device)
    n_total = 0

    for start in range(0, X_t.shape[0], batch_size):
        x = X_t[start:start + batch_size]
        b = B[start:start + batch_size]
        dx = x - b
        integral_grads = torch.zeros_like(x)
        prev_grad = None
        for alpha in alphas:
            x_interp = (b + alpha * dx).detach().requires_grad_(True)
            z = encoder(x_interp)
            z_std = (z - mean) / (std + latent_std_epsilon)
            z_pred = generator(z_std)
            diff = z_pred - z_std
            score = 0.5 * (diff * diff).sum(dim=1).mean()
            grad = torch.autograd.grad(score, x_interp, retain_graph=False, create_graph=False)[0]
            if prev_grad is not None:
                integral_grads += 0.5 * (prev_grad + grad) * delta
            prev_grad = grad
        ig = dx * integral_grads
        attr_sum_abs += ig.abs().sum(dim=0)
        n_total += x.shape[0]

    vals = (attr_sum_abs / max(n_total, 1)).detach().cpu().numpy()
    return pd.DataFrame({"gene": np.asarray(gene_names).astype(str), "IG_mean_abs": vals}).sort_values(
        "IG_mean_abs", ascending=False
    ).reset_index(drop=True)


def historical_driver_score(
    X_ctrl, X_real, X_pred, gene_names, ig_df, eps=1e-8, *,
    real_abs_weight=1.00, predicted_abs_weight=0.40,
    good_match_weight=0.75, sign_match_weight=0.35, shift_error_weight=-0.40,
    good_match_abs_pred_threshold=0.1,
):
    """DriverScore calculation exactly as used by the notebook IG sections."""
    def mean_expr(X):
        return np.log1p(np.expm1(np.asarray(X, np.float32)).mean(axis=0) + eps)

    mu_ctrl, mu_real, mu_pred = mean_expr(X_ctrl), mean_expr(X_real), mean_expr(X_pred)
    lfc_real = np.log2((mu_real + eps) / (mu_ctrl + eps))
    lfc_pred = np.log2((mu_pred + eps) / (mu_ctrl + eps))
    abs_real, abs_pred = np.abs(lfc_real), np.abs(lfc_pred)
    sign_match = (np.sign(lfc_real) == np.sign(lfc_pred)).astype(float)
    good_match = ((abs_pred > good_match_abs_pred_threshold) & (sign_match > 0)).astype(float)
    shift_error = np.abs(abs_real - abs_pred)
    df = pd.DataFrame({
        "gene": np.asarray(gene_names).astype(str),
        "lfc_real": lfc_real, "lfc_pred": lfc_pred,
        "abs_lfc_real": abs_real, "abs_lfc_pred": abs_pred,
        "sign_match": sign_match, "good_match": good_match,
        "shift_error": shift_error,
    })
    df["DriverScore"] = (
        real_abs_weight * df["abs_lfc_real"]
        + predicted_abs_weight * df["abs_lfc_pred"]
        + good_match_weight * df["good_match"]
        + sign_match_weight * df["sign_match"]
        + shift_error_weight * df["shift_error"]
    )
    df = df.merge(ig_df[["gene", "IG_mean_abs"]], on="gene", how="inner")
    lo, hi = df.IG_mean_abs.min(), df.IG_mean_abs.max()
    df["IG_abs_norm"] = 0.0 if hi - lo < 1e-12 else (df.IG_mean_abs - lo) / (hi - lo)
    std = np.nanstd(df["DriverScore"].to_numpy())
    if std > 1e-12:
        df["DriverScore_z"] = (df["DriverScore"] - np.nanmean(df["DriverScore"])) / (std + 1e-12)
    else:
        df["DriverScore_z"] = 0.0
    return df.sort_values("DriverScore", ascending=False).reset_index(drop=True)


def filter_and_categorize(
    df, X_ctrl, X_real, gene_names, *, threshold=1e-6, min_det=0.10, lfc_clip=6.0, q=0.75,
    real_abs_weight=1.00, predicted_abs_weight=0.40,
    good_match_weight=0.75, sign_match_weight=0.35, shift_error_weight=-0.40,
):
    det_ctrl = (np.asarray(X_ctrl) > threshold).mean(axis=0)
    det_real = (np.asarray(X_real) > threshold).mean(axis=0)
    passing = set(np.asarray(gene_names).astype(str)[(det_ctrl >= min_det) & (det_real >= min_det)])
    out = df[df.gene.isin(passing)].copy()
    for col in ["lfc_real", "lfc_pred"]:
        out[col] = out[col].clip(-lfc_clip, lfc_clip)
    out["abs_lfc_real"] = out.lfc_real.abs(); out["abs_lfc_pred"] = out.lfc_pred.abs()
    out["shift_error"] = (out.abs_lfc_real - out.abs_lfc_pred).abs()
    out["DriverScore"] = (
        real_abs_weight * out.abs_lfc_real
        + predicted_abs_weight * out.abs_lfc_pred
        + good_match_weight * out.good_match
        + sign_match_weight * out.sign_match
        + shift_error_weight * out.shift_error
    )
    drv_thr = out.DriverScore.quantile(q); ig_thr = out.IG_abs_norm.quantile(q)
    def category(row):
        hd, hi = row.DriverScore >= drv_thr, row.IG_abs_norm >= ig_thr
        if hd and hi: return "Key Perturbation Genes"
        if hi: return "Sensitive Regulators"
        if hd: return "Large Movers"
        return "Low Both"
    out["Category"] = out.apply(category, axis=1)
    return out.sort_values("DriverScore", ascending=False), float(drv_thr), float(ig_thr)



def plot_driver_summary(merged, drv_thr, ig_thr, gene, *, top_n=40, label_k=20, save_path=None, show=False):
    """Reproduce the two-panel driver-gene summary used in the notebook.

    Mitochondrial and ribosomal genes are removed from the visualization only,
    matching the original plotting function; they remain in the saved full
    DriverScore table.
    """
    g = merged["gene"].astype(str).str.upper()
    rm_mask = (
        g.str.startswith("MT-") | g.str.startswith("RPL") |
        g.str.startswith("RPS") | g.str.startswith("MRPL") |
        g.str.startswith("MRPS")
    )
    df = merged[~rm_mask].copy()
    high_drv = df[df["DriverScore"] >= drv_thr]
    high_both = df[(df["DriverScore"] >= drv_thr) & (df["IG_abs_norm"] >= ig_thr)]
    corr = float(np.corrcoef(df["DriverScore"].values, df["IG_abs_norm"].values)[0, 1])

    to_label = (
        pd.concat([
            high_both.sort_values("DriverScore", ascending=False),
            df.sort_values("DriverScore", ascending=False),
        ])
        .drop_duplicates("gene")
        .head(label_k)
    )

    fig = plt.figure(figsize=(16, 7), dpi=150)
    gs = GridSpec(1, 2, width_ratios=[1.3, 1.0], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    hb = ax0.hexbin(df["IG_abs_norm"], df["DriverScore"], gridsize=55, mincnt=1, cmap="YlOrRd")
    fig.colorbar(hb, ax=ax0).set_label("Gene density")
    ax0.scatter(high_drv["IG_abs_norm"], high_drv["DriverScore"], s=18, alpha=0.5, color="#4C72B0", label="Large Movers")
    ax0.scatter(high_both["IG_abs_norm"], high_both["DriverScore"], s=45, alpha=0.9, color="#FF7F0E", zorder=5, label="Key Perturbation Genes")
    ax0.axhline(drv_thr, linestyle="--", lw=1, color="gray")
    ax0.axvline(ig_thr, linestyle="--", lw=1, color="gray")
    for _, row in to_label.iterrows():
        ax0.annotate(row["gene"], (row["IG_abs_norm"], row["DriverScore"]), fontsize=8, xytext=(3, 3), textcoords="offset points", alpha=0.9)
    ax0.set_xlabel("IG_abs_norm  (model sensitivity to input gene)", fontsize=11)
    ax0.set_ylabel("DriverScore  (biological perturbation effect)", fontsize=11)
    ax0.set_title(f"ctrl → {gene} Driver Genes\nDriverScore vs IG  r={corr:.3f}", fontsize=12)
    ax0.legend(frameon=False, fontsize=9)

    ax1 = fig.add_subplot(gs[0, 1])
    top = df.nlargest(top_n, "DriverScore").sort_values("IG_abs_norm", ascending=True)
    y = np.arange(len(top))
    ax1.barh(y, top["IG_abs_norm"], color="#4C72B0", alpha=0.8, label="IG (model)")
    ax1.set_yticks(y)
    ax1.set_yticklabels(top["gene"], fontsize=9)
    ax1.set_xlabel("IG_abs_norm", fontsize=10)
    ax1.set_title(f"Top {top_n} Perturbation Drivers — IG Sensitivity", fontsize=11)
    ax1b = ax1.twiny()
    ax1b.scatter(top["DriverScore"], y, s=22, color="#FF7F0E", alpha=0.9, label="DriverScore (biology)", zorder=5)
    ax1b.set_xlabel("DriverScore", color="#FF7F0E", fontsize=10)
    ax1b.tick_params(axis="x", colors="#FF7F0E")
    fig.suptitle(f"Cell Translator: ctrl → {gene} Perturbation Driver Genes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"driver_ig_correlation": corr, "n_key_genes": int(len(high_both)), "n_mito_ribo_removed_for_plot": int(rm_mask.sum())}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", required=True, choices=["APOE", "MSI2", "NR1H3", "LRRC4C", "ANKRD12"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", default=None, help="translated_validation_outputs.npz; defaults to the configured output directory")
    parser.add_argument("--show-plots", action="store_true", help="Display figures interactively in addition to saving them")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["gene"] != args.gene:
        raise ValueError("Gene/config mismatch")

    import tensorflow as tf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_ctrl = np.load(cfg["data"]["control_expression"]).astype(np.float32)
    X_real = np.load(cfg["data"]["target_expression"]).astype(np.float32)
    genes = np.load(cfg["data"]["gene_names"], allow_pickle=False)
    pred_path = args.predictions or str(Path(cfg["outputs"]["directory"]) / "translated_validation_outputs.npz")
    pred = np.load(pred_path, allow_pickle=False)
    src_idx = pred["source_validation_indices"].astype(int)
    tgt_idx = pred["target_validation_indices"].astype(int)
    X_pred = pred["X_pred_hard"].astype(np.float32)

    lat = load_perturbation_latents(cfg["archived_artifacts"]["ae_outputs"], args.gene)
    ae, _ = load_perturbation_ae_checkpoint(
        cfg["archived_artifacts"]["ae_checkpoint"], args.gene, X_ctrl.shape[1], device
    )
    keras_G = tf.keras.models.load_model(cfg["archived_artifacts"]["forward_generator"], compile=False)
    G = load_keras_generator_into_torch(keras_G, TorchForwardGenerator(lat["train_src"].shape[1])).to(device)

    ig_cfg = cfg["interpretation"]["integrated_gradients"]
    ig = integrated_gradients_latent_shift(
        ae.encoder_ctrl, G, X_ctrl[src_idx], genes, lat["mean_ctrl"], lat["std_ctrl"],
        steps=int(ig_cfg["steps"]), max_samples=int(ig_cfg["max_samples"]),
        baseline_mode=ig_cfg["baseline_mode"], batch_size=int(ig_cfg["batch_size"]),
        seed=int(ig_cfg["seed"]),
        latent_std_epsilon=float(cfg["implementation_constants"]["latent_standardization_epsilon"]),
        device=device,
    )
    drv_cfg = cfg["interpretation"]["driver_score"]
    score_weights = {
        "real_abs_weight": float(drv_cfg["real_abs_weight"]),
        "predicted_abs_weight": float(drv_cfg["predicted_abs_weight"]),
        "good_match_weight": float(drv_cfg["good_match_weight"]),
        "sign_match_weight": float(drv_cfg["sign_match_weight"]),
        "shift_error_weight": float(drv_cfg["shift_error_weight"]),
        "good_match_abs_pred_threshold": float(drv_cfg["good_match_abs_pred_threshold"]),
    }
    merged = historical_driver_score(
        X_ctrl[src_idx], X_real[tgt_idx], X_pred, genes, ig,
        eps=float(drv_cfg["epsilon"]), **score_weights
    )
    filter_weights = {
        key: value for key, value in score_weights.items()
        if key != "good_match_abs_pred_threshold"
    }
    filtered, drv_thr, ig_thr = filter_and_categorize(
        merged, X_ctrl[src_idx], X_real[tgt_idx], genes,
        threshold=float(cfg["evaluation"]["threshold"]),
        min_det=float(drv_cfg["min_detection"]), lfc_clip=float(drv_cfg["lfc_clip"]),
        q=float(drv_cfg["quantile"]), **filter_weights,
    )
    out = Path(cfg["outputs"]["directory"]) / "interpretation"
    out.mkdir(parents=True, exist_ok=True)
    ig.to_csv(out / "integrated_gradients.csv", index=False)
    filtered.to_csv(out / "driver_scores.csv", index=False)
    plot_stats = plot_driver_summary(
        filtered, drv_thr, ig_thr, args.gene,
        top_n=int(drv_cfg["top_n"]), label_k=int(drv_cfg["label_k"]),
        save_path=out / "driver_summary.png", show=args.show_plots,
    )
    (out / "thresholds.txt").write_text(
        f"DriverScore_q={drv_thr}\nIG_q={ig_thr}\n"
        f"DriverScore_IG_r={plot_stats['driver_ig_correlation']}\n"
        f"n_key_genes={plot_stats['n_key_genes']}\n"
        f"n_mito_ribo_removed_for_plot={plot_stats['n_mito_ribo_removed_for_plot']}\n"
    )


if __name__ == "__main__":
    main()
