"""Training utilities for the independent-state HurdleAE variant.

This module preserves the HurdleAE loss and training procedure used for the
acute leukemia CD34+, BMMC, and MPAL state models.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from cell_translator.models.independent_hurdle_ae import IndependentHurdleAutoencoder


@dataclass(frozen=True)
class IndependentHurdleAETrainingConfig:
    latent_dim: int = 128
    dropout: float = 0.10
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 200
    patience: int = 10
    min_delta: float = 1e-4
    validation_fraction: float = 0.20
    seed: int = 42
    input_drop_prob: float = 0.15
    threshold: float = 0.0
    lambda_bce: float = 1.0
    lambda_nz: float = 1.0
    lambda_fp: float = 2.0
    huber_beta: float = 0.1
    grad_clip: float = 1.0
    p_zero_penalty_weight: float = 0.0
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4


class NumpyExpressionDataset(Dataset):
    def __init__(self, array_2d: np.ndarray):
        x = np.asarray(array_2d, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D expression matrix, got {x.shape}.")
        if not np.isfinite(x).all():
            raise ValueError("Expression matrix contains NaN or Inf values.")
        if (x < 0).any():
            raise ValueError("Acute leukemia HurdleAE inputs must be non-negative log1p values.")
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]


def corrupt_nonzero_entries(x: torch.Tensor, drop_prob: float = 0.10) -> torch.Tensor:
    if drop_prob <= 0:
        return x
    nz = x > 0
    drop = (torch.rand_like(x) < drop_prob) & nz
    x2 = x.clone()
    x2[drop] = 0.0
    return x2


def independent_hurdle_loss(
    x_hat: torch.Tensor,
    logits: torch.Tensor,
    x_true: torch.Tensor,
    threshold: float = 0.0,
    lambda_bce: float = 1.0,
    lambda_nz: float = 1.0,
    lambda_fp: float = 2.0,
    huber_beta: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Historical acute-leukemia HurdleAE objective."""
    mask = (x_true > threshold).float()
    bce = nn.BCEWithLogitsLoss()(logits, mask)

    nz_count = mask.sum() + eps
    l1 = nn.SmoothL1Loss(reduction="none", beta=huber_beta)(x_hat, x_true)
    l1_nz = (l1 * mask).sum() / nz_count

    zmask = 1.0 - mask
    z_count = zmask.sum() + eps
    fp = ((x_hat * zmask) ** 2).sum() / z_count
    return lambda_bce * bce + lambda_nz * l1_nz + lambda_fp * fp


def train_independent_hurdle_ae(model, train_dl, val_dl, device, config: IndependentHurdleAETrainingConfig):
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0
    train_losses, val_losses, val_latent_vars = [], [], []

    for ep in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for x in train_dl:
            x = x.to(device)
            x_in = corrupt_nonzero_entries(x, config.input_drop_prob)
            z, x_hat, logits, p, mag = model(x_in)
            loss = independent_hurdle_loss(
                x_hat, logits, x,
                threshold=config.threshold,
                lambda_bce=config.lambda_bce,
                lambda_nz=config.lambda_nz,
                lambda_fp=config.lambda_fp,
                huber_beta=config.huber_beta,
            )
            if config.p_zero_penalty_weight > 0:
                mask = (x > config.threshold).float()
                zmask = 1.0 - mask
                loss = loss + config.p_zero_penalty_weight * (p * zmask).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()
            total += float(loss.item())
        train_loss = total / max(len(train_dl), 1)
        train_losses.append(train_loss)

        model.eval()
        total = 0.0
        zvars = []
        with torch.no_grad():
            for x in val_dl:
                x = x.to(device)
                z, x_hat, logits, p, mag = model(x)
                loss = independent_hurdle_loss(
                    x_hat, logits, x,
                    threshold=config.threshold,
                    lambda_bce=config.lambda_bce,
                    lambda_nz=config.lambda_nz,
                    lambda_fp=config.lambda_fp,
                    huber_beta=config.huber_beta,
                )
                if config.p_zero_penalty_weight > 0:
                    mask = (x > config.threshold).float()
                    zmask = 1.0 - mask
                    loss = loss + config.p_zero_penalty_weight * (p * zmask).mean()
                total += float(loss.item())
                # Historical notebook used torch.var's default unbiased=True.
                zvars.append(float(z.var(dim=0).mean().item()))

        val_loss = total / max(len(val_dl), 1)
        val_losses.append(val_loss)
        val_latent_vars.append(float(np.mean(zvars)) if zvars else 0.0)
        sched.step(val_loss)

        print(
            f"Epoch {ep:03d} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"latent_var={val_latent_vars[-1]:.6f}"
        )
        if val_loss < best_val - config.min_delta:
            best_val = val_loss
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                print(
                    f"Early stopping at epoch {ep}. Best epoch: {best_epoch} | "
                    f"best val={best_val:.4f}"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train_losses": np.asarray(train_losses, dtype=np.float32),
        "val_losses": np.asarray(val_losses, dtype=np.float32),
        "val_latent_vars": np.asarray(val_latent_vars, dtype=np.float32),
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
    }


def infer_independent_hurdle(model, X: np.ndarray, device, batch_size: int = 256):
    ds = NumpyExpressionDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    zs, xs, ps, mags = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            z, x_hat, logits, p, mag = model(x)
            zs.append(z.cpu().numpy())
            xs.append(x_hat.cpu().numpy())
            ps.append(p.cpu().numpy())
            mags.append(mag.cpu().numpy())
    return tuple(np.concatenate(v, axis=0).astype(np.float32) for v in (zs, xs, ps, mags))


def decode_independent_latent(model, Z: np.ndarray, device, batch_size: int = 512):
    Z = np.asarray(Z, dtype=np.float32)
    xs, ps, mags = [], [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, Z.shape[0], batch_size):
            z = torch.as_tensor(Z[start:start + batch_size], dtype=torch.float32, device=device)
            x_hat, logits, p, mag = model.decode(z)
            xs.append(x_hat.cpu().numpy())
            ps.append(p.cpu().numpy())
            mags.append(mag.cpu().numpy())
    return tuple(np.concatenate(v, axis=0).astype(np.float32) for v in (xs, ps, mags))


def run_independent_hurdle_ae(
    npy_path: str,
    save_dir: str,
    run_name: str,
    config: IndependentHurdleAETrainingConfig,
):
    """Train one state-specific HurdleAE and save checkpoint/latent arrays."""
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    X = np.load(npy_path).astype(np.float32)
    _ = NumpyExpressionDataset(X)  # validate before splitting
    idx = np.arange(X.shape[0])
    train_idx, val_idx = train_test_split(
        idx,
        test_size=config.validation_fraction,
        random_state=config.seed,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IndependentHurdleAutoencoder(
        X.shape[1], latent_dim=config.latent_dim, dropout=config.dropout
    ).to(device)
    train_dl = DataLoader(
        NumpyExpressionDataset(X[train_idx]), batch_size=config.batch_size,
        shuffle=True, drop_last=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_dl = DataLoader(
        NumpyExpressionDataset(X[val_idx]), batch_size=config.batch_size,
        shuffle=False, drop_last=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    history = train_independent_hurdle_ae(model, train_dl, val_dl, device, config)
    Z_all, X_expected, P_all, MAG_all = infer_independent_hurdle(
        model, X, device, config.batch_size
    )

    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / f"{run_name}_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {**asdict(config), "input_dim": int(X.shape[1])},
            "history": history,
            "train_idx": train_idx.astype(np.int64),
            "val_idx": val_idx.astype(np.int64),
        },
        checkpoint_path,
    )
    latent_path = out / f"{run_name}_latents.npz"
    np.savez_compressed(
        latent_path,
        Z_all=Z_all,
        Z_train=Z_all[train_idx],
        Z_val=Z_all[val_idx],
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        X_expected_all=X_expected,
        P_all=P_all,
        MAG_all=MAG_all,
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved latents: {latent_path}")
    return {
        "X": X, "model": model, "device": device,
        "train_idx": train_idx, "val_idx": val_idx,
        "Z_all": Z_all, "history": history,
        "checkpoint_path": str(checkpoint_path),
        "latents_npz_path": str(latent_path),
    }


def calibrate_global_p_threshold(P: np.ndarray, target_detection_fraction: float) -> float:
    return float(np.quantile(np.asarray(P).ravel(), 1.0 - target_detection_fraction))
