"""Optional Integrated Gradients and DriverScore analysis for morphology translation.

This module preserves the historical PyTorch mirror used to attribute the
Keras morphology generator. It reports the Keras/PyTorch forward discrepancy
before computing attributions so the approximation is explicit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_translator.training.morphology_ae_training import (
    load_morphology_ae,
    load_state_artifacts,
    inverse_scale,
)
from config_utils import load_config


class TorchMorphologyGenerator(nn.Module):
    """Historical PyTorch mirror of the forward Keras morphology generator."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 256)
        self.ln3 = nn.LayerNorm(256)
        self.fc4 = nn.Linear(256, latent_dim)
        self.head = nn.Linear(latent_dim, latent_dim)

    def forward(self, z):
        z = self.ln1(F.relu(self.fc1(z)))
        z = self.ln2(F.relu(self.fc2(z)))
        z = self.ln3(F.relu(self.fc3(z)))
        z = self.fc4(z)
        return self.head(z)


def _collect_dense_ln(model):
    import tensorflow as tf
    layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layers.extend(_collect_dense_ln(layer))
        elif isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.LayerNormalization)):
            layers.append(layer)
    return layers


def load_keras_generator_into_torch(keras_model, torch_model):
    import tensorflow as tf
    all_layers = _collect_dense_ln(keras_model)
    dense = [x for x in all_layers if isinstance(x, tf.keras.layers.Dense)]
    norms = [x for x in all_layers if isinstance(x, tf.keras.layers.LayerNormalization)]
    if len(dense) < 5 or len(norms) < 3:
        raise ValueError(f"Expected at least 5 Dense and 3 LayerNormalization layers; found {len(dense)} and {len(norms)}")

    def set_linear(target, source):
        weight, bias = source.get_weights()
        target.weight.data = torch.tensor(weight.T, dtype=torch.float32)
        target.bias.data = torch.tensor(bias, dtype=torch.float32)

    def set_norm(target, source):
        gamma, beta = source.get_weights()
        target.weight.data = torch.tensor(gamma, dtype=torch.float32)
        target.bias.data = torch.tensor(beta, dtype=torch.float32)

    set_linear(torch_model.fc1, dense[0]); set_norm(torch_model.ln1, norms[0])
    set_linear(torch_model.fc2, dense[1]); set_norm(torch_model.ln2, norms[1])
    set_linear(torch_model.fc3, dense[2]); set_norm(torch_model.ln3, norms[2])
    set_linear(torch_model.fc4, dense[3]); set_linear(torch_model.head, dense[4])
    return torch_model


class MorphologyTranslator(nn.Module):
    def __init__(self, source_ae, generator, mean_source, std_source):
        super().__init__()
        self.encoder = source_ae.encoder
        self.generator = generator
        self.register_buffer("mean_lat", torch.tensor(np.asarray(mean_source, np.float32)))
        self.register_buffer("std_lat", torch.tensor(np.asarray(std_source, np.float32)))

    def forward(self, x):
        z = self.encoder(x)
        z_source_std = (z - self.mean_lat) / (self.std_lat + 1e-6)
        z_target_std = self.generator(z_source_std)
        return z_source_std, z_target_std


def integrated_gradients_translation(translator, X_source_scaled, feature_names, steps=64, max_samples=1024, batch_size=16, seed=42, device=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    translator.eval()
    X = np.asarray(X_source_scaled, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        X = X[rng.choice(len(X), max_samples, replace=False)]

    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    baseline = X_t.mean(dim=0, keepdim=True).repeat(X_t.shape[0], 1)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=device)
    delta = 1.0 / steps
    attr_sum = torch.zeros(X_t.shape[1], device=device)
    n_total = 0

    for start in range(0, len(X_t), batch_size):
        x = X_t[start:start + batch_size]
        b = baseline[start:start + batch_size]
        dx = x - b
        integral = torch.zeros_like(x)
        previous = None
        for alpha in alphas:
            x_interp = (b + alpha * dx).detach().requires_grad_(True)
            z_src, z_tgt = translator(x_interp)
            score = 0.5 * ((z_tgt - z_src) ** 2).sum(dim=1).mean()
            grad = torch.autograd.grad(score, x_interp, retain_graph=False, create_graph=False)[0]
            if previous is not None:
                integral += 0.5 * (previous + grad) * delta
            previous = grad
        attr_sum += (dx * integral).abs().sum(dim=0)
        n_total += len(x)

    values = (attr_sum / max(n_total, 1)).detach().cpu().numpy()
    return pd.DataFrame({"feature": list(feature_names), "IG_mean_abs": values}).sort_values("IG_mean_abs", ascending=False).reset_index(drop=True)


def compute_driver_score(X_source, X_target, X_translated, feature_names, ig_df, variance_tol=1e-6):
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    X_translated = np.asarray(X_translated, dtype=np.float64)
    mu_s, mu_t, mu_f = np.nanmean(X_source, 0), np.nanmean(X_target, 0), np.nanmean(X_translated, 0)
    n_s, n_t = len(X_source), len(X_target)
    sd_s, sd_t = np.nanstd(X_source, 0, ddof=1), np.nanstd(X_target, 0, ddof=1)
    pooled = np.sqrt((((n_s - 1) * sd_s**2) + ((n_t - 1) * sd_t**2)) / (n_s + n_t - 2))
    finite = pooled[np.isfinite(pooled) & (pooled > 0)]
    relative_floor = np.nanmedian(finite) * 1e-4 if finite.size else variance_tol
    sd_floor = max(variance_tol, relative_floor)
    valid = np.isfinite(mu_s) & np.isfinite(mu_t) & np.isfinite(mu_f) & np.isfinite(pooled) & (pooled > sd_floor)

    d_real = np.full(len(feature_names), np.nan)
    d_fake = np.full(len(feature_names), np.nan)
    d_real[valid] = (mu_t[valid] - mu_s[valid]) / pooled[valid]
    d_fake[valid] = (mu_f[valid] - mu_s[valid]) / pooled[valid]
    out = pd.DataFrame({"feature": feature_names, "cohens_d_real": d_real, "cohens_d_fake": d_fake, "valid_variance": valid})
    out = out.loc[out.valid_variance].copy()
    out["DriverScore"] = np.abs(out["cohens_d_real"])
    out["sign_match"] = (np.sign(out["cohens_d_real"]) == np.sign(out["cohens_d_fake"])).astype(float)
    out["shift_error"] = np.abs(out["cohens_d_real"] - out["cohens_d_fake"])
    out = out.merge(ig_df[["feature", "IG_mean_abs"]], on="feature", how="inner")
    mn, mx = out["IG_mean_abs"].min(), out["IG_mean_abs"].max()
    out["IG_abs_norm"] = (out["IG_mean_abs"] - mn) / (mx - mn + 1e-12)
    return out.sort_values("DriverScore", ascending=False).reset_index(drop=True)


def categorize_features(df, quantile=0.75):
    driver_threshold = df["DriverScore"].quantile(quantile)
    ig_threshold = df["IG_abs_norm"].quantile(quantile)
    out = df.copy()
    high_driver = out["DriverScore"] >= driver_threshold
    high_ig = out["IG_abs_norm"] >= ig_threshold
    out["Category"] = "Low Both"
    out.loc[high_driver & ~high_ig, "Category"] = "Large Movers"
    out.loc[~high_driver & high_ig, "Category"] = "Sensitive Regulators"
    out.loc[high_driver & high_ig, "Category"] = "Key Translation Features"
    return out


def run_interpretation(cfg, transition_label):
    import tensorflow as tf
    transition = next(x for x in cfg["transitions"] if x["label"] == transition_label)
    source, target = transition["source"], transition["target"]
    artifact_dir = cfg["paths"]["output_dir"]
    run_dir = Path(artifact_dir) / f"{source}_to_{target}"
    generator_path = run_dir / "generator_forward.keras"
    generator = tf.keras.models.load_model(generator_path, compile=False)

    source_art = load_state_artifacts(source, artifact_dir)
    target_art = load_state_artifacts(target, artifact_dir)
    source_model, _ = load_morphology_ae(source, artifact_dir)
    device = next(source_model.parameters()).device
    latent_dim = source_art["Z_train"].shape[1]
    mirror = TorchMorphologyGenerator(latent_dim).to(device)
    load_keras_generator_into_torch(generator, mirror)
    mirror.eval()

    mean_src = source_art["Z_train"].mean(0, keepdims=True).astype(np.float32)
    std_src = np.maximum(source_art["Z_train"].std(0, keepdims=True), 1e-6).astype(np.float32)

    # Historical mirror fidelity diagnostic.
    rng = np.random.default_rng(42)
    z_check = rng.normal(size=(16, latent_dim)).astype(np.float32)
    keras_out = generator(z_check, training=False).numpy()
    with torch.no_grad():
        torch_out = mirror(torch.tensor(z_check, device=device)).cpu().numpy()
    max_diff = float(np.max(np.abs(keras_out - torch_out)))
    print(f"Keras/PyTorch generator mirror max difference: {max_diff:.6f}")

    translator = MorphologyTranslator(source_model, mirror, mean_src, std_src).to(device)
    features = source_art["scaler_stats"]["feature"].astype(str).tolist()
    ig = integrated_gradients_translation(
        translator,
        source_art["X_validation_scaled"],
        features,
        steps=int(cfg["interpretation"]["ig_steps"]),
        max_samples=int(cfg["interpretation"]["max_samples"]),
        batch_size=int(cfg["interpretation"]["batch_size"]),
        seed=int(cfg["interpretation"]["seed"]),
        device=device,
    )
    ig.to_csv(run_dir / "integrated_gradients.csv", index=False)

    source_original = inverse_scale(source_art["X_validation_scaled"], source_art["scaler_stats"])
    target_original = inverse_scale(target_art["X_validation_scaled"], target_art["scaler_stats"])
    translated = np.load(run_dir / "translated_features.npy")
    driver = compute_driver_score(source_original, target_original, translated, features, ig)
    driver = categorize_features(driver, float(cfg["interpretation"]["category_quantile"]))
    driver.to_csv(run_dir / "driver_features.csv", index=False)
    print(driver["Category"].value_counts())
    return max_diff, driver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--transition", required=True)
    args = parser.parse_args()
    run_interpretation(load_config(args.config), args.transition)


if __name__ == "__main__":
    main()
