"""Integrated-gradients and DriverScore analysis for the primary organoid run.

This reproduces the final coherent 128-dimensional driver-gene workflow in the
source notebook. Earlier exploratory 250-dimensional latent-axis cells are not
part of this module.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

from cell_translator.models.hurdle_ae import HurdleAutoencoder
from config_utils import load_config


class TorchForwardGenerator(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, 1024); self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512); self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 512); self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, d); self.head = nn.Linear(d, d)
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
    def dense_copy(pt, kr):
        w, b = kr.get_weights(); pt.weight.data = torch.tensor(w.T, dtype=torch.float32); pt.bias.data = torch.tensor(b, dtype=torch.float32)
    def ln_copy(pt, kr):
        g, b = kr.get_weights(); pt.weight.data = torch.tensor(g, dtype=torch.float32); pt.bias.data = torch.tensor(b, dtype=torch.float32)
    dense_copy(g_torch.fc1, dense[0]); ln_copy(g_torch.ln1, norms[0])
    dense_copy(g_torch.fc2, dense[1]); ln_copy(g_torch.ln2, norms[1])
    dense_copy(g_torch.fc3, dense[2]); ln_copy(g_torch.ln3, norms[2])
    dense_copy(g_torch.fc4, dense[3]); dense_copy(g_torch.head, dense[4])
    return g_torch


class LatentTranslator(nn.Module):
    def __init__(self, ae, generator, mean_lat, std_lat):
        super().__init__()
        self.encoder = ae.encoder_net
        self.generator = generator
        self.register_buffer("mean_lat", torch.tensor(np.asarray(mean_lat, np.float32)))
        self.register_buffer("std_lat", torch.tensor(np.asarray(std_lat, np.float32)))


def integrated_gradients(translator, x_np, gene_names, steps=64, max_samples=1024, batch_size=16, seed=42, device=None):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    x_np = np.asarray(x_np, np.float32)
    if x_np.shape[0] > max_samples:
        x_np = x_np[rng.choice(x_np.shape[0], max_samples, replace=False)]
    x_all = torch.as_tensor(x_np, device=device)
    baseline = x_all.mean(0, keepdim=True).repeat(x_all.shape[0], 1)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)
    delta = 1.0 / steps
    attr_sum = torch.zeros(x_all.shape[1], device=device)
    total = 0
    for start in range(0, x_all.shape[0], batch_size):
        x = x_all[start:start+batch_size]; b = baseline[start:start+batch_size]; dx = x-b
        integral = torch.zeros_like(x); prev = None
        for a in alphas:
            xi = (b + a*dx).detach().requires_grad_(True)
            z = translator.encoder(xi)
            z_std = (z-translator.mean_lat)/(translator.std_lat+1e-6)
            z_fake = translator.generator(z_std)
            score = 0.5*((z_fake-z_std)**2).sum(1).mean()
            grad = torch.autograd.grad(score, xi)[0]
            if prev is not None: integral += 0.5*(prev+grad)*delta
            prev = grad
        attr_sum += (dx*integral).abs().sum(0); total += x.shape[0]
    values = (attr_sum/max(total,1)).detach().cpu().numpy()
    return pd.DataFrame({"gene": gene_names, "IG_mean_abs": values}).sort_values("IG_mean_abs", ascending=False).reset_index(drop=True)


def safe_mean_log1p(x, eps=1e-8):
    return np.log1p(np.expm1(np.asarray(x, np.float32)).mean(0) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not cfg.get("interpretability", {}).get("enabled", False):
        raise SystemExit("Interpretability is not enabled for this dataset configuration.")

    prepared = Path(cfg["paths"]["prepared_dir"]); out = Path(cfg["paths"]["output_dir"])
    src = np.load(prepared / f"{cfg['source_state']}_X_hvg.npy").astype(np.float32)
    genes = np.load(prepared / "hvg_genes.npy", allow_pickle=True).astype(str)
    ae_npz = np.load(out / f"{cfg['run_name']}_outputs.npz", allow_pickle=False)
    translated = np.load(out / f"{cfg['source_state'].lower()}_to_{cfg['target_state'].lower()}" / "translated_source_outputs.npz", allow_pickle=False)["X_translated"]

    # Target validation cells, matching the translated source-validation output.
    names = [str(x) for x in ae_npz["state_names"].tolist()]
    starts = ae_npz["slice_starts"].astype(int); ends = ae_npz["slice_ends"].astype(int)
    slices = {n:(s,e) for n,s,e in zip(names,starts,ends)}
    s,e = slices[cfg["target_state"]]; val = np.intersect1d(np.arange(s,e), ae_npz["val_idx"]) - s
    target_all = np.load(prepared / f"{cfg['target_state']}_X_hvg.npy").astype(np.float32)
    target_val = target_all[val]

    ckpt_path = out / f"{cfg['run_name']}_best.pt"
    try: ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError: ckpt = torch.load(ckpt_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = HurdleAutoencoder(src.shape[1], int(ckpt["config"]["latent_dim"]), float(ckpt["config"]["dropout"])).to(device)
    ae.load_state_dict(ckpt["model_state_dict"]); ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)

    g_keras = tf.keras.models.load_model(out / f"{cfg['source_state'].lower()}_to_{cfg['target_state'].lower()}" / "G_forward.keras", compile=False)
    g_torch = load_keras_into_torch(g_keras, TorchForwardGenerator(int(cfg["model"]["latent_dim"]))).to(device).eval()
    for p in g_torch.parameters(): p.requires_grad_(False)
    translator = LatentTranslator(ae, g_torch, ae_npz["mean_z_train"], ae_npz["std_z_train"]).to(device).eval()

    icfg = cfg["interpretability"]
    ig = integrated_gradients(translator, src, genes, steps=int(icfg["ig_steps"]), max_samples=int(icfg["ig_max_samples"]), batch_size=int(icfg["ig_batch_size"]), seed=int(icfg["seed"]), device=device)
    ig.to_csv(out / "integrated_gradients.csv", index=False)

    eps = float(icfg["driver_eps"]); clip = float(icfg["lfc_clip"])
    mu_src = safe_mean_log1p(src, eps); mu_real = safe_mean_log1p(target_val, eps); mu_fake = safe_mean_log1p(translated, eps)
    lfc_real = np.clip(np.log2((mu_real+eps)/(mu_src+eps)), -clip, clip)
    lfc_fake = np.clip(np.log2((mu_fake+eps)/(mu_src+eps)), -clip, clip)
    df = pd.DataFrame({"gene":genes,"log2FC_real":lfc_real,"log2FC_fake":lfc_fake,"abs_log2FC_real":np.abs(lfc_real),"abs_log2FC_fake":np.abs(lfc_fake)})
    df["sign_match"] = (np.sign(lfc_real)==np.sign(lfc_fake)).astype(float)
    df["good_match"] = ((np.abs(lfc_fake)>0.1)&(df["sign_match"]>0)).astype(float)
    df["shift_error"] = np.abs(np.abs(lfc_real)-np.abs(lfc_fake)); df["DriverScore"] = np.abs(lfc_real)
    df = df.merge(ig, on="gene", how="inner")
    df["IG_abs_norm"] = (df.IG_mean_abs-df.IG_mean_abs.min())/(df.IG_mean_abs.max()-df.IG_mean_abs.min()+1e-12)

    det_src = (src > float(icfg["detection_threshold"])).mean(0)
    det_tgt = (target_val > float(icfg["detection_threshold"])).mean(0)
    passing = set(genes[(det_src >= float(icfg["min_detection_fraction"])) & (det_tgt >= float(icfg["min_detection_fraction"]))])
    df = df[df.gene.isin(passing)].copy().reset_index(drop=True)
    q = float(icfg["category_quantile"]); dthr=df.DriverScore.quantile(q); ithr=df.IG_abs_norm.quantile(q)
    def cat(r):
        hd=r.DriverScore>=dthr; hi=r.IG_abs_norm>=ithr
        return "Key Translation Genes" if hd and hi else "Sensitive Regulators" if hi else "Large Movers" if hd else "Low Both"
    df["Category"] = df.apply(cat, axis=1)
    df.to_csv(out / "driver_genes.csv", index=False)
    print(f"Genes passing detection filter: {len(df)} / {len(genes)}")
    print(df.Category.value_counts())
    print(df[df.Category=="Key Translation Genes"].head(15).to_string(index=False))

if __name__ == "__main__":
    main()
