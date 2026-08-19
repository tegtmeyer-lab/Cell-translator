"""Integrated Gradients and DriverScore analysis for NGN3 to Endocrine translation."""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from scipy.stats import pearsonr

from cell_translator.models.hurdle_ae import HurdleAutoencoder
from config_utils import load_config


class TorchForwardGenerator(nn.Module):
    """PyTorch mirror of the saved Keras forward generator."""
    def __init__(self, d: int):
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


def _collect_dense_ln(model):
    out = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            out.extend(_collect_dense_ln(layer))
        elif isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.LayerNormalization)):
            out.append(layer)
    return out


def load_keras_into_torch(g_keras, g_torch):
    layers = _collect_dense_ln(g_keras)
    dense = [x for x in layers if isinstance(x, tf.keras.layers.Dense)]
    norms = [x for x in layers if isinstance(x, tf.keras.layers.LayerNormalization)]
    if len(dense) != 5 or len(norms) != 3:
        raise ValueError("Unexpected forward-generator architecture.")

    def copy_dense(pt, kr):
        w, b = kr.get_weights()
        pt.weight.data = torch.tensor(w.T, dtype=torch.float32)
        pt.bias.data = torch.tensor(b, dtype=torch.float32)

    def copy_ln(pt, kr):
        g, b = kr.get_weights()
        pt.weight.data = torch.tensor(g, dtype=torch.float32)
        pt.bias.data = torch.tensor(b, dtype=torch.float32)

    copy_dense(g_torch.fc1, dense[0]); copy_ln(g_torch.ln1, norms[0])
    copy_dense(g_torch.fc2, dense[1]); copy_ln(g_torch.ln2, norms[1])
    copy_dense(g_torch.fc3, dense[2]); copy_ln(g_torch.ln3, norms[2])
    copy_dense(g_torch.fc4, dense[3]); copy_dense(g_torch.head, dense[4])
    return g_torch


class LatentTranslator(nn.Module):
    def __init__(self, ae, generator, mean_lat, std_lat):
        super().__init__()
        self.encoder = ae.encoder_net
        self.generator = generator
        self.register_buffer("mean_lat", torch.tensor(np.asarray(mean_lat, np.float32)))
        self.register_buffer("std_lat", torch.tensor(np.asarray(std_lat, np.float32)))


def integrated_gradients(translator, x_np, genes, steps=64, max_samples=1024, batch_size=16, seed=42, device=None):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    x_np = np.asarray(x_np, np.float32)
    if x_np.shape[0] > max_samples:
        x_np = x_np[rng.choice(x_np.shape[0], max_samples, replace=False)]
    x_all = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    baseline = x_all.mean(0, keepdim=True).repeat(x_all.shape[0], 1)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)
    delta = 1.0 / steps
    attr_sum = torch.zeros(x_all.shape[1], device=device)
    total = 0

    for start in range(0, x_all.shape[0], batch_size):
        x = x_all[start:start+batch_size]
        b = baseline[start:start+batch_size]
        dx = x-b
        integral = torch.zeros_like(x)
        previous = None
        for alpha in alphas:
            xi = (b + alpha*dx).detach().requires_grad_(True)
            z = translator.encoder(xi)
            z_std = (z-translator.mean_lat)/(translator.std_lat+1e-6)
            z_fake = translator.generator(z_std)
            score = 0.5*((z_fake-z_std)**2).sum(1).mean()
            grad = torch.autograd.grad(score, xi)[0]
            if previous is not None:
                integral += 0.5*(previous+grad)*delta
            previous = grad
        attr_sum += (dx*integral).abs().sum(0)
        total += x.shape[0]

    values = (attr_sum/max(total,1)).detach().cpu().numpy()
    return pd.DataFrame({"gene": genes, "IG_mean_abs": values}).sort_values("IG_mean_abs", ascending=False).reset_index(drop=True)


def safe_mean_log1p(x, eps=1e-8):
    # Historical notebook definition, retained exactly.
    return np.log1p(np.expm1(np.asarray(x, np.float32)).mean(0) + eps)


def _target_validation(ae, target_all, target_state):
    names = [str(x) for x in ae["state_names"].tolist()]
    slices = {n:(int(s),int(e)) for n,s,e in zip(names, ae["slice_starts"], ae["slice_ends"])}
    s,e = slices[target_state]
    val_idx = ae["val_idx"] if "val_idx" in ae.files else ae["test_idx"]
    local = np.intersect1d(np.arange(s,e), val_idx) - s
    return target_all[local]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    icfg = cfg["interpretability"]

    prepared = Path(cfg["paths"]["prepared_dir"])
    out = Path(cfg["paths"]["output_dir"])
    source = np.load(prepared / f"{cfg['source_state']}_X_hvg.npy").astype(np.float32)
    target_all = np.load(prepared / f"{cfg['target_state']}_X_hvg.npy").astype(np.float32)
    genes = np.load(prepared / "hvg_genes.npy", allow_pickle=False).astype(str)

    ae_npz = np.load(out / f"{cfg['run_name']}_outputs.npz", allow_pickle=False)
    target_val = _target_validation(ae_npz, target_all, cfg["target_state"])
    slug = f"{cfg['source_state'].lower()}_to_{cfg['target_state'].lower()}".replace(" ", "_")
    translated = np.load(out / slug / "translated_source_outputs.npz", allow_pickle=False)["X_translated"].astype(np.float32)

    ckpt_path = out / f"{cfg['run_name']}_best.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae_model = HurdleAutoencoder(
        source.shape[1], int(ckpt["config"]["latent_dim"]), float(ckpt["config"]["dropout"])
    ).to(device)
    ae_model.load_state_dict(ckpt["model_state_dict"])
    ae_model.eval()
    for p in ae_model.parameters():
        p.requires_grad_(False)

    g_keras = tf.keras.models.load_model(out / slug / "G_forward.keras", compile=False)
    g_torch = load_keras_into_torch(
        g_keras, TorchForwardGenerator(int(cfg["model"]["latent_dim"]))
    ).to(device).eval()
    for p in g_torch.parameters():
        p.requires_grad_(False)
    translator = LatentTranslator(
        ae_model, g_torch, ae_npz["mean_z_train"], ae_npz["std_z_train"]
    ).to(device).eval()

    ig = integrated_gradients(
        translator, source, genes,
        steps=int(icfg["ig_steps"]),
        max_samples=int(icfg["ig_max_samples"]),
        batch_size=int(icfg["ig_batch_size"]),
        seed=int(icfg["seed"]),
        device=device,
    )
    ig.to_csv(out / "integrated_gradients.csv", index=False)
    print(f"Top IG gene: {ig.iloc[0]['gene']} ({ig.iloc[0]['IG_mean_abs']:.6f})")

    eps = float(icfg["driver_eps"])
    cap = float(icfg["lfc_clip"])
    mu_src = safe_mean_log1p(source, eps)
    mu_real = safe_mean_log1p(target_val, eps)
    mu_fake = safe_mean_log1p(translated, eps)
    real_fc = np.clip(np.log2((mu_real+eps)/(mu_src+eps)), -cap, cap)
    fake_fc = np.clip(np.log2((mu_fake+eps)/(mu_src+eps)), -cap, cap)

    df = pd.DataFrame({
        "gene": genes,
        "log2FC_real": real_fc,
        "log2FC_fake": fake_fc,
        "abs_log2FC_real": np.abs(real_fc),
        "abs_log2FC_fake": np.abs(fake_fc),
    })
    df["sign_match"] = (np.sign(real_fc) == np.sign(fake_fc)).astype(float)
    df["good_match"] = ((np.abs(fake_fc)>0.1) & (df["sign_match"]>0)).astype(float)
    df["shift_error"] = np.abs(np.abs(real_fc)-np.abs(fake_fc))
    df["DriverScore"] = np.abs(real_fc)
    df = df.merge(ig, on="gene", how="inner")
    df["IG_abs_norm"] = (
        (df["IG_mean_abs"]-df["IG_mean_abs"].min())
        /(df["IG_mean_abs"].max()-df["IG_mean_abs"].min()+1e-12)
    )
    df.to_csv(out / "driver_genes_full.csv", index=False)

    thr = float(icfg["detection_threshold"])
    det_src = (source > thr).mean(0)
    det_tgt = (target_val > thr).mean(0)
    passing = set(genes[
        (det_src >= float(icfg["min_detection_fraction"]))
        & (det_tgt >= float(icfg["min_detection_fraction"]))
    ])
    df = df[df["gene"].isin(passing)].copy().reset_index(drop=True)

    q = float(icfg["category_quantile"])
    dthr = df["DriverScore"].quantile(q)
    ithr = df["IG_abs_norm"].quantile(q)
    def category(row):
        high_d = row["DriverScore"] >= dthr
        high_i = row["IG_abs_norm"] >= ithr
        if high_d and high_i: return "Key Translation Genes"
        if high_i: return "Sensitive Regulators"
        if high_d: return "Large Movers"
        return "Low Both"
    df["Category"] = df.apply(category, axis=1)
    df.to_csv(out / "driver_genes.csv", index=False)

    print(f"Genes passing detection filter: {len(df)} / {len(genes)}")
    print(df["Category"].value_counts())
    print(df[df["Category"] == "Key Translation Genes"].head(15).to_string(index=False))

    # Historical post-filter summary after removing mito/ribosomal prefixes.
    upper = df["gene"].astype(str).str.upper()
    remove = (
        upper.str.startswith("MT-") | upper.str.startswith("MT_")
        | upper.str.startswith("RPL") | upper.str.startswith("RPS")
        | upper.str.startswith("MRPL") | upper.str.startswith("MRPS")
    )
    plot_df = df[~remove].copy()
    r, p = pearsonr(plot_df["DriverScore"], plot_df["IG_abs_norm"])
    print(f"Removed {int(remove.sum())} mito/ribo genes. Remaining: {len(plot_df)}")
    print(f"DriverScore vs IG_abs_norm: r={r:.3f}, p={p:.3e}")
    print(
        "High Driver:", int((plot_df["DriverScore"] >= dthr).sum()),
        "| High IG:", int((plot_df["IG_abs_norm"] >= ithr).sum()),
        "| Intersection:", int(
            ((plot_df["DriverScore"] >= dthr) & (plot_df["IG_abs_norm"] >= ithr)).sum()
        ),
    )


if __name__ == "__main__":
    main()
