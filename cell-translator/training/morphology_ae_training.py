"""Training and artifact utilities for the Cell Painting morphology autoencoder.

The implementation preserves the architecture, loss terms, optimization
settings, train/validation split, and train-only imputation/scaling procedure
used in the reported morphology analyses.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from cell_translator.models.morphology_ae import MorphologyAutoencoder


class NumpyMorphologyDataset(Dataset):
    def __init__(self, array):
        self.data = torch.as_tensor(np.asarray(array, dtype=np.float32))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]


@dataclass(frozen=True)
class MorphologyAETrainingConfig:
    latent_dim: int = 256
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-4
    validation_fraction: float = 0.2
    split_random_state: int = 42
    patience: int = 20
    lambda_variance: float = 2.0
    lambda_covariance: float = 0.1
    weight_decay: float = 1e-5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 8
    gradient_clip: float = 1.0
    early_stopping_min_delta: float = 1e-6
    inference_batch_size: int = 256
    # The source notebook did not seed PyTorch before AE initialization/training.
    torch_seed: int | None = None

    @classmethod
    def from_dict(cls, values: dict | None):
        values = dict(values or {})
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in values.items() if k in allowed})


def morphology_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    lambda_variance: float = 2.0,
    lambda_covariance: float = 0.1,
):
    """MSE + feature-SD matching + reconstruction off-diagonal covariance penalty."""
    mse = F.mse_loss(recon, target)
    std_real = target.std(dim=0) + 1e-8
    std_recon = recon.std(dim=0) + 1e-8
    variance_loss = F.mse_loss(std_recon, std_real)

    recon_centered = recon - recon.mean(dim=0, keepdim=True)
    covariance = (recon_centered.T @ recon_centered) / (recon.shape[0] - 1)
    off_diag = covariance - torch.diag(torch.diag(covariance))
    covariance_loss = (off_diag ** 2).mean()
    return mse + lambda_variance * variance_loss + lambda_covariance * covariance_loss


def train_morphology_ae(model, train_loader, validation_dataset, device, config):
    """Train one morphology AE using the historical training/validation objectives."""
    if config.torch_seed is not None:
        torch.manual_seed(int(config.torch_seed))
        np.random.seed(int(config.torch_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.torch_seed))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    best_val, best_state, best_epoch, wait = float("inf"), None, -1, 0
    train_losses, val_losses, latent_vars = [], [], []
    validation_data = validation_dataset.data.to(device)

    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            _, recon = model(batch)
            loss = morphology_loss(
                recon,
                batch,
                lambda_variance=config.lambda_variance,
                lambda_covariance=config.lambda_covariance,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip)
            optimizer.step()
            total += float(loss.item())

        train_loss = total / max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            z_val, recon_val = model(validation_data)
            # The notebook schedules/stops on validation MSE only.
            val_loss = float(F.mse_loss(recon_val, validation_data).item())
            latent_var = float(z_val.var(dim=0).mean().item())

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        latent_vars.append(latent_var)

        print(
            f"Epoch {epoch:03d}/{config.epochs} | train={train_loss:.5f} | "
            f"val={val_loss:.5f} | latent_var={latent_var:.5f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val - config.early_stopping_min_delta:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            wait += 1
            if wait >= config.patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch {best_epoch} | val={best_val:.5f}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "latent_vars": latent_vars,
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
    }


def _infer(model, X, device, batch_size, mode):
    loader = DataLoader(
        NumpyMorphologyDataset(X),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    outputs = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if mode == "encode":
                out = model.encode(batch)
            elif mode == "decode":
                out = model.decode(batch)
            else:
                _, out = model(batch)
            outputs.append(out.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def get_latents(model, X, device, batch_size=256):
    return _infer(model, X, device, batch_size, "encode")


def decode_latents(model, Z, device, batch_size=256):
    return _infer(model, Z, device, batch_size, "decode")


def get_reconstructions(model, X, device, batch_size=256):
    return _infer(model, X, device, batch_size, "reconstruct")


def reconstruction_metrics(X_real_scaled, X_recon_scaled, scaler):
    """Reconstruction metrics matching the notebook's held-out evaluation."""
    X_real = scaler.inverse_transform(X_real_scaled)
    X_recon = scaler.inverse_transform(X_recon_scaled)

    mse_cells = ((X_recon_scaled - X_real_scaled) ** 2).mean(axis=1)
    mse_features = ((X_recon_scaled - X_real_scaled) ** 2).mean(axis=0)

    ss_res = ((X_real - X_recon) ** 2).sum(axis=0)
    ss_tot = ((X_real - X_real.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)

    r_feature = pearsonr(X_real.mean(0), X_recon.mean(0))[0]
    r_std = pearsonr(X_real.std(0), X_recon.std(0))[0]

    n_sample = min(X_real.shape[0], 5000)
    rng = np.random.default_rng(0)
    idx = rng.choice(X_real.shape[0], n_sample, replace=False)
    r_overall = pearsonr(X_real[idx].ravel(), X_recon[idx].ravel())[0]

    return {
        "median_cell_mse": float(np.median(mse_cells)),
        "median_feature_mse": float(np.median(mse_features)),
        "median_r2": float(np.median(r2)),
        "frac_r2_gt_0_5": float((r2 > 0.5).mean()),
        "feature_mean_pearson_r": float(r_feature),
        "feature_std_pearson_r": float(r_std),
        "overall_value_pearson_r": float(r_overall),
    }


def run_morphology_ae_state(X_raw, feature_names: Sequence[str], state_name: str, save_dir: str, config):
    """Fit one state-specific AE with train-only median imputation/scaling."""
    X_df = pd.DataFrame(np.asarray(X_raw), columns=list(feature_names))
    n_cells = len(X_df)

    train_idx, val_idx = train_test_split(
        np.arange(n_cells),
        test_size=config.validation_fraction,
        random_state=config.split_random_state,
        shuffle=True,
    )

    train_median = X_df.iloc[train_idx].median()
    X_imputed = X_df.fillna(train_median)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_imputed.iloc[train_idx]).astype(np.float32)
    X_val = scaler.transform(X_imputed.iloc[val_idx]).astype(np.float32)
    X_all = scaler.transform(X_imputed).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MorphologyAutoencoder(
        input_dim=X_train.shape[1], latent_dim=config.latent_dim
    ).to(device)

    train_loader = DataLoader(
        NumpyMorphologyDataset(X_train),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    history = train_morphology_ae(
        model, train_loader, NumpyMorphologyDataset(X_val), device, config
    )

    Z_train = get_latents(model, X_train, device, config.inference_batch_size)
    Z_val = get_latents(model, X_val, device, config.inference_batch_size)
    Z_all = get_latents(model, X_all, device, config.inference_batch_size)
    R_train = get_reconstructions(model, X_train, device, config.inference_batch_size)
    R_val = get_reconstructions(model, X_val, device, config.inference_batch_size)

    metrics_train = reconstruction_metrics(X_train, R_train, scaler)
    metrics_val = reconstruction_metrics(X_val, R_val, scaler)

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / state_name

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "input_dim": int(X_train.shape[1]),
            "feature_names": list(feature_names),
            "history": history,
            "train_idx": train_idx.astype(np.int64),
            "validation_idx": val_idx.astype(np.int64),
        },
        str(prefix) + "_ae_checkpoint.pt",
    )

    arrays = {
        "Z_train": Z_train,
        "Z_validation": Z_val,
        "Z_all": Z_all,
        "X_train_scaled": X_train,
        "X_validation_scaled": X_val,
        "X_all_scaled": X_all,
        "train_idx": train_idx.astype(np.int64),
        "validation_idx": val_idx.astype(np.int64),
    }
    for suffix, value in arrays.items():
        np.save(str(prefix) + f"_{suffix}.npy", value)

    pd.DataFrame(
        {
            "feature": list(feature_names),
            "train_median": train_median.values,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        }
    ).to_csv(str(prefix) + "_scaler_stats.csv", index=False)

    metrics = {
        "state": state_name,
        "n_cells": int(n_cells),
        "best_epoch": history["best_epoch"],
        "best_val": history["best_val"],
        "train": metrics_train,
        "validation": metrics_val,
    }
    with open(str(prefix) + "_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def load_morphology_ae(state_name: str, artifact_dir: str, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        Path(artifact_dir) / f"{state_name}_ae_checkpoint.pt", map_location=device
    )
    model = MorphologyAutoencoder(
        input_dim=int(checkpoint["input_dim"]),
        latent_dim=int(checkpoint["config"]["latent_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_state_artifacts(state_name: str, artifact_dir: str):
    root = Path(artifact_dir)
    return {
        "Z_train": np.load(root / f"{state_name}_Z_train.npy"),
        "Z_validation": np.load(root / f"{state_name}_Z_validation.npy"),
        "X_train_scaled": np.load(root / f"{state_name}_X_train_scaled.npy"),
        "X_validation_scaled": np.load(root / f"{state_name}_X_validation_scaled.npy"),
        "train_idx": np.load(root / f"{state_name}_train_idx.npy"),
        "validation_idx": np.load(root / f"{state_name}_validation_idx.npy"),
        "scaler_stats": pd.read_csv(root / f"{state_name}_scaler_stats.csv"),
    }


def inverse_scale(X_scaled, scaler_stats: pd.DataFrame):
    mean = scaler_stats["scaler_mean"].to_numpy(dtype=np.float64)
    scale = scaler_stats["scaler_scale"].to_numpy(dtype=np.float64)
    return (np.asarray(X_scaled, dtype=np.float64) * scale + mean).astype(np.float32)
