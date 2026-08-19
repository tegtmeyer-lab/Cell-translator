"""HurdleAE loss, training, calibration, and shared-run orchestration.

The scientific implementation is extracted from the reviewer-ready reference
script; only module imports were reorganized.
"""

import os
import copy
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from cell_translator.hyperparameters import *
from cell_translator.models.hurdle_ae import HurdleAutoencoder
from cell_translator.utils.reproducibility import set_torch_seed
from cell_translator.utils.validation import (
    validate_expression_matrix, basic_stats, check_mag_scale,
    check_latent_separation, check_hard_decode_quality,
)

def corrupt_nonzero_entries(x: torch.Tensor, drop_prob: float = 0.05) -> torch.Tensor:
    """Denoising: randomly zero out some expressed entries during training."""
    if drop_prob <= 0:
        return x
    nz = (x > 0)
    drop = (torch.rand_like(x) < drop_prob) & nz
    x2 = x.clone()
    x2[drop] = 0.0
    return x2

class NumpyExpressionDataset(Dataset):
    def __init__(self, array_2d: np.ndarray):
        arr = np.asarray(array_2d, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("Input contains NaN/Inf.")
        self.x = torch.from_numpy(arr)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]

def compute_pos_weight_per_gene(
    X_train: np.ndarray, thr: float = 1e-6, clip_max: float = 50.0
) -> torch.Tensor:
    """Per-gene BCE pos_weight = neg_count / pos_count (handles class imbalance)."""
    X = np.asarray(X_train, dtype=np.float32)
    pos = (X > thr).sum(axis=0).astype(np.float32)
    neg = X.shape[0] - pos
    pos = np.maximum(pos, 1.0)
    pw = np.clip(neg / pos, 1.0, clip_max).astype(np.float32)
    return torch.from_numpy(pw)

def masked_smooth_l1(
    pred, target, mask, beta=0.1, alpha_tail=1.0, tail_clip=6.0, eps=1e-8
):
    """SmoothL1 loss computed only on nonzero entries, with tail upweighting."""
    l = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    w_tail = 1.0 + alpha_tail * torch.clamp(target, max=tail_clip)
    l = l * w_tail
    denom = mask.sum().clamp_min(1.0)
    return (l * mask).sum() / (denom + eps)

def hurdle_loss(
    logits, p, mag, x_true,
    pos_weight_gene,
    thr=1e-6,
    lambda_bce=1.0,
    lambda_mag=5.0,
    lambda_fp=2.0,
    lambda_soft=0.5,
    huber_beta=0.1,
    alpha_tail=1.0,
    tail_clip=6.0,
):
    mask = (x_true > thr).float()    # 1 where expressed
    zmask = 1.0 - mask                # 1 where zero

    # 1. BCE: learn which genes are expressed (weighted per gene)
    bce = F.binary_cross_entropy_with_logits(
        logits, mask,
        pos_weight=pos_weight_gene.view(1, -1),
        reduction="mean"
    )

    # 2. Magnitude fidelity: mag should match expression on nonzeros
    mag_nz = masked_smooth_l1(
        mag, x_true, mask,
        beta=huber_beta, alpha_tail=alpha_tail, tail_clip=tail_clip
    )

    # 3. False positive penalty: push p LOW on true zeros
    fp = (p * zmask).sum() / zmask.sum().clamp_min(1.0)

    # 4. Soft reconstruction stabilizer: p*mag should approximate x_true globally
    x_soft = p * mag
    soft = F.smooth_l1_loss(x_soft, x_true, reduction="mean", beta=huber_beta)

    return lambda_bce * bce + lambda_mag * mag_nz + lambda_fp * fp + lambda_soft * soft

def train_shared_hurdle_ae(
    model, train_dl, val_dl, device,
    pos_weight_gene,
    epochs=300,
    lr=5e-4,
    patience=25,
    min_delta=1e-4,
    input_drop_prob=0.05,
    thr=1e-6,
    lambda_bce=1.0,
    lambda_mag=5.0,
    lambda_fp=2.0,
    lambda_soft=0.5,
    huber_beta=0.1,
    alpha_tail=1.0,
    tail_clip=6.0,
    grad_clip=1.0,
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0

    pos_weight_gene = pos_weight_gene.to(device)
    train_losses, val_losses, latent_vars = [], [], []

    for ep in range(1, epochs + 1):
        # ---- train ----
        model.train()
        t = 0.0
        for x in train_dl:
            x = x.to(device, non_blocking=True)
            x_in = corrupt_nonzero_entries(x, drop_prob=input_drop_prob)

            z, x_soft, logits, p, mag = model(x_in)
            loss = hurdle_loss(
                logits=logits, p=p, mag=mag, x_true=x,
                pos_weight_gene=pos_weight_gene, thr=thr,
                lambda_bce=lambda_bce, lambda_mag=lambda_mag,
                lambda_fp=lambda_fp, lambda_soft=lambda_soft,
                huber_beta=huber_beta, alpha_tail=alpha_tail, tail_clip=tail_clip,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            t += float(loss.item())

        train_loss = t / max(len(train_dl), 1)
        train_losses.append(train_loss)

        # ---- validate ----
        model.eval()
        v, zvars = 0.0, []
        with torch.no_grad():
            for x in val_dl:
                x = x.to(device, non_blocking=True)
                z, x_soft, logits, p, mag = model(x)
                loss = hurdle_loss(
                    logits=logits, p=p, mag=mag, x_true=x,
                    pos_weight_gene=pos_weight_gene, thr=thr,
                    lambda_bce=lambda_bce, lambda_mag=lambda_mag,
                    lambda_fp=lambda_fp, lambda_soft=lambda_soft,
                    huber_beta=huber_beta, alpha_tail=alpha_tail, tail_clip=tail_clip,
                )
                v += float(loss.item())
                zvars.append(float(z.var(dim=0, unbiased=False).mean().item()))

        val_loss = v / max(len(val_dl), 1)
        latent_var = float(np.mean(zvars)) if zvars else 0.0
        val_losses.append(val_loss)
        latent_vars.append(latent_var)

        sched.step(val_loss)
        lr_now = opt.param_groups[0]["lr"]
        print(
            f"Epoch {ep:03d} | train={train_loss:.4f} | val={val_loss:.4f} | "
            f"latent_var={latent_var:.6f} | lr={lr_now:.2e}"
        )

        # ---- early stopping ----
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep}. Best epoch {best_epoch} | val={best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": np.array(train_losses, dtype=np.float32),
        "val_losses": np.array(val_losses, dtype=np.float32),
        "latent_vars": np.array(latent_vars, dtype=np.float32),
        "best_epoch": int(best_epoch),
        "best_val": float(best_val),
    }

def infer_z_p_mag(model, X: np.ndarray, device, batch_size: int = 256):
    ds = NumpyExpressionDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    Zs, Ps, MAGs = [], [], []
    model.eval()
    with torch.no_grad():
        for x in dl:
            x = x.to(device, non_blocking=True)
            z, x_soft, logits, p, mag = model(x)
            Zs.append(z.cpu().numpy())
            Ps.append(p.cpu().numpy())
            MAGs.append(mag.cpu().numpy())

    Z = np.concatenate(Zs, axis=0).astype(np.float32)
    P = np.concatenate(Ps, axis=0).astype(np.float32)
    MAG = np.concatenate(MAGs, axis=0).astype(np.float32)
    return Z, P, MAG

def calibrate_tau_gene(
    P_train: np.ndarray, X_train: np.ndarray, thr: float = 1e-6
) -> np.ndarray:
    """
    For each gene g, set tau_gene[g] = quantile(P_train[:,g], p0_real[g])
    so that hard decoding reproduces the observed zero fraction per gene.
    """
    p0_real = (np.asarray(X_train, np.float32) <= thr).mean(axis=0)  # (G,)
    n_genes = P_train.shape[1]
    tau_gene = np.array([
        np.quantile(P_train[:, g], float(p0_real[g]))
        for g in range(n_genes)
    ], dtype=np.float32)
    return tau_gene

def hard_decode(P: np.ndarray, MAG: np.ndarray, tau_gene: np.ndarray) -> np.ndarray:
    gate = (P > tau_gene.reshape(1, -1)).astype(np.float32)
    return (gate * MAG).astype(np.float32)

def z_standardize_stats(Z_train: np.ndarray, eps: float = 1e-6):
    mean = Z_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(Z_train.std(axis=0, keepdims=True), eps).astype(np.float32)
    return mean, std

def run_shared_hurdle_ae(
    x_paths: Dict[str, str],
    state_names: List[str],
    save_dir: str,
    run_name: str,
    latent_dim: int,
) -> Dict[str, Any]:
    """
    Train a shared HurdleAE across the given cell states and save its
    latents/checkpoint for downstream CycleGAN translation.

    This single function replaces the notebook's two near-duplicate
    orchestration functions (one for the joint 4-state run, one for the
    dedicated 2-state NPC/Astrocyte run) -- they were identical except for
    which states/paths/latent_dim were used, so that varying configuration
    is now passed in as arguments instead of being copy-pasted.
    """
    set_torch_seed(SEED)
    os.makedirs(save_dir, exist_ok=True)
    out_prefix = os.path.join(save_dir, run_name)

    # ---- load, validate, and concatenate all states ----
    X_list, y_list, slices = [], [], {}
    start = 0
    expected_n_genes = None
    for k in state_names:
        if k not in x_paths:
            raise KeyError(f"Missing input path for state '{k}'.")
        if not os.path.isfile(x_paths[k]):
            raise FileNotFoundError(f"Input file not found for {k}: {x_paths[k]}")

        Xk = validate_expression_matrix(
            np.load(x_paths[k]),
            name=k,
            expected_n_genes=expected_n_genes,
        )
        if expected_n_genes is None:
            expected_n_genes = Xk.shape[1]

        X_list.append(Xk)
        # Store labels as Unicode strings so the NPZ can be loaded without pickle.
        y_list.append(np.full(Xk.shape[0], fill_value=k, dtype=f"<U{max(1, len(k))}"))
        slices[k] = (start, start + Xk.shape[0])
        start += Xk.shape[0]
        print(f"Loaded {k}: {Xk.shape}")

    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0).astype(str)
    basic_stats(X_all, "X_all", thr=THR)

    # ---- stratified train/validation split ----
    # The 20% subset is used for HurdleAE early stopping and is therefore
    # correctly termed validation rather than an independent test set.
    idx_all = np.arange(X_all.shape[0])
    tr_idx, val_idx = train_test_split(
        idx_all, test_size=0.2, random_state=SEED, shuffle=True, stratify=y_all
    )
    X_train, X_val = X_all[tr_idx], X_all[val_idx]
    basic_stats(X_train, "X_train", thr=THR)
    basic_stats(X_val, "X_val", thr=THR)

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HurdleAutoencoder(
        input_dim=X_all.shape[1], latent_dim=latent_dim, dropout=DROPOUT
    ).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    pos_weight_gene = compute_pos_weight_per_gene(X_train, thr=THR, clip_max=50.0)

    # ---- loaders ----
    train_dl = DataLoader(
        NumpyExpressionDataset(X_train),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_dl = DataLoader(
        NumpyExpressionDataset(X_val),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    # ---- train ----
    hist = train_shared_hurdle_ae(
        model=model, train_dl=train_dl, val_dl=val_dl,
        device=device, pos_weight_gene=pos_weight_gene,
        epochs=EPOCHS, lr=LR, patience=PATIENCE, min_delta=MIN_DELTA,
        input_drop_prob=INPUT_DROP_PROB, thr=THR,
        lambda_bce=LAMBDA_BCE, lambda_mag=LAMBDA_MAG,
        lambda_fp=LAMBDA_FP, lambda_soft=LAMBDA_SOFT,
        huber_beta=HUBER_BETA, alpha_tail=ALPHA_TAIL, tail_clip=TAIL_CLIP,
    )

    # ---- infer on all cells ----
    Z_all, P_all, MAG_all = infer_z_p_mag(model, X_all, device, batch_size=256)
    assert Z_all.shape[0] == X_all.shape[0], \
        f"Shape mismatch: Z_all={Z_all.shape[0]} vs X_all={X_all.shape[0]}"

    Z_train_lat = Z_all[tr_idx]
    Z_val_lat = Z_all[val_idx]

    # ---- sanity checks ----
    check_mag_scale(MAG_all, X_all, thr=THR)
    check_latent_separation(Z_all, slices, state_names)

    # ---- standardization stats for CycleGAN ----
    mean_z, std_z = z_standardize_stats(Z_train_lat)
    np.save(out_prefix + "_mean_z_train.npy", mean_z)
    np.save(out_prefix + "_std_z_train.npy", std_z)

    # ---- per-state tau calibration + hard decode quality check ----
    tau_gene_states = {}
    hard_states_stats = {}

    for k in state_names:
        s, e = slices[k]
        idx_state = np.arange(s, e)
        idx_state_tr = np.intersect1d(idx_state, tr_idx)

        if idx_state_tr.size < 10:
            print(f"[WARN] {k}: too few train cells ({idx_state_tr.size}) for tau calibration")
            continue

        tau_k = calibrate_tau_gene(P_all[idx_state_tr], X_all[idx_state_tr], thr=THR)
        tau_gene_states[k] = tau_k

        X_hard_k = hard_decode(P_all[idx_state], MAG_all[idx_state], tau_k)
        check_hard_decode_quality(X_hard_k, X_all[idx_state], name=k, thr=THR)

        hard_states_stats[k] = {
            "zero_frac": float((X_hard_k <= THR).mean()),
            "median_gene_zero": float(np.median((X_hard_k <= THR).mean(axis=0))),
            "median_rowsum": float(np.median(X_hard_k.sum(axis=1))),
        }

        np.save(out_prefix + f"_tau_gene_{k}.npy", tau_k)

    # ---- save checkpoint ----
    ckpt_path = out_prefix + "_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "datasets": x_paths, "seed": SEED,
            "input_dim": X_all.shape[1], "latent_dim": latent_dim, "dropout": DROPOUT,
            "batch_size": BATCH_SIZE, "lr": LR,
            "epochs": EPOCHS, "patience": PATIENCE, "thr": THR,
            "loss": {
                "lambda_bce": LAMBDA_BCE, "lambda_mag": LAMBDA_MAG,
                "lambda_fp": LAMBDA_FP, "lambda_soft": LAMBDA_SOFT,
                "huber_beta": HUBER_BETA, "alpha_tail": ALPHA_TAIL, "tail_clip": TAIL_CLIP,
            }
        },
        "history": hist,
        "train_idx": tr_idx.astype(np.int64),
        "val_idx": val_idx.astype(np.int64),
        "slices": slices,
        "state_names": state_names,
    }, ckpt_path)
    print(f"\nSaved checkpoint: {ckpt_path}")

    # ---- save outputs ----
    out_npz = out_prefix + "_outputs.npz"
    slice_starts = np.array([slices[k][0] for k in state_names], dtype=np.int64)
    slice_ends = np.array([slices[k][1] for k in state_names], dtype=np.int64)
    np.savez_compressed(
        out_npz,
        Z_all=Z_all, Z_train=Z_train_lat, Z_val=Z_val_lat,
        P_all=P_all, MAG_all=MAG_all,
        y_all=y_all,
        train_idx=tr_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        state_names=np.asarray(state_names, dtype=str),
        slice_starts=slice_starts, slice_ends=slice_ends,
        mean_z_train=mean_z, std_z_train=std_z,
    )
    print(f"Saved outputs: {out_npz}")

    return {
        "X_all": X_all, "y_all": y_all,
        "model": model, "device": device,
        "history": hist,
        "train_idx": tr_idx, "val_idx": val_idx,
        "slices": slices,
        "Z_all": Z_all, "P_all": P_all, "MAG_all": MAG_all,
        "tau_gene_states": tau_gene_states,
        "hard_states_stats": hard_states_stats,
        "mean_z_train": mean_z, "std_z_train": std_z,
        "checkpoint_path": ckpt_path,
        "outputs_npz_path": out_npz,
    }
