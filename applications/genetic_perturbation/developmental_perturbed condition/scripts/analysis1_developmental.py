# %% [markdown]
# # Cell Translator — developmental NPC --> Neuron translation across conditions
#  
# Configured transitions:
# 
# 1. Control NPC -> Control Neuron
# 2. APOE-perturbed NPC -> APOE-perturbed Neuron
# 3. MSI2-perturbed NPC -> MSI2-perturbed Neuron
# 4. NR1H3-perturbed NPC -> NR1H3-perturbed Neuron
# 5. ANKRD12-perturbed NPC -> ANKRD12-perturbed Neuron
# 6. LRRC4C-perturbed NPC -> LRRC4C-perturbed Neuron
# 
# ## Design choices
# 
# - One pairwise HurdleAE + latent CycleGAN is trained **per condition**.
# - HurdleAE domains are now **source NPC** and **target Neuron**.
# - All CSVs are restricted to one common gene set and one fixed gene order.
# - NPC and Neuron cells are split independently into **70% train / 15% validation / 15% test**.
# - All AE latent standardization and hurdle thresholds are estimated from **training cells only**.
# - CycleGAN is trained only on training latents. Validation latents are used for monitoring. Test cells are untouched until final evaluation.
# - Final metrics are computed only on held-out test cells.
# - If donor/cell-line/biological-replicate labels are available, the split function supports **group-held-out splitting**; this is preferable to cell-level random splitting for biological generalization.

# %%
# ============================================================
# 1. Imports and configuration
# ============================================================
from pathlib import Path
import os
import gc
import json
import copy
import math
import random
import warnings
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense, LayerNormalization
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from scipy.stats import pearsonr, spearmanr, ks_2samp, wasserstein_distance
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42


# Portable configuration. Paths in config.json are relative to repository root
# unless absolute; no dataset paths are embedded in this source file.
import argparse as _argparse
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ARGP = _argparse.ArgumentParser(description='Cell Translator developmental workflow')
_ARGP.add_argument('--config', type=Path, default=_REPO_ROOT / 'configs' / 'config.json')
_ARGP.add_argument('--all', action='store_true', help='Run every condition/fold')
_ARGP.add_argument('--aggregate-only', action='store_true', help='Aggregate existing results without training')
_ARGP.add_argument('--condition', choices=['control','APOE','MSI2','NR1H3','ANKRD12','LRRC4C'])

_ARGS = _ARGP.parse_args()
if not _ARGS.config.exists():
    raise FileNotFoundError(
        f'Configuration not found: {_ARGS.config}. '
        'Copy configs/config.example.json to configs/config.json and edit input paths.'
    )
with open(_ARGS.config, encoding='utf-8') as _f:
    _CONFIG = json.load(_f)

def _resolve_path(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (_REPO_ROOT / path).resolve())

INPUT_FILES = {
    cond: {state: _resolve_path(path) for state, path in states.items()}
    for cond, states in _CONFIG['input_files'].items()
}
CONDITIONS = list(_CONFIG.get('conditions', ['control','APOE','MSI2','NR1H3','ANKRD12','LRRC4C']))

OUTPUT_ROOT = Path(_resolve_path(_CONFIG.get('output_dir', 'outputs/developmental')))
RUN_CONDITIONS = CONDITIONS
GROUP_COLUMN = _CONFIG.get('group_column', 'group')
GROUP_FILES = {
    cond: {state: _resolve_path(_CONFIG.get('group_files', {}).get(cond, {}).get(state))
           for state in ('NPC', 'Neuron')}
    for cond in RUN_CONDITIONS
}
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-8

# -----------------------------
# HurdleAE hyperparameters
# Preserved from the uploaded notebook unless noted.
# -----------------------------
LATENT_DIM = 128
AE_DROPOUT = 0.10
AE_BATCH_SIZE = 64
AE_LR = 5e-4
AE_EPOCHS = 300
AE_PATIENCE = 25
AE_MIN_DELTA = 1e-4

THR = 1e-6
INPUT_DROP_PROB = 0.05
LAMBDA_BCE  = 1.0
LAMBDA_MAG  = 5.0
LAMBDA_FP   = 2.0
LAMBDA_SOFT = 0.5
HUBER_BETA  = 0.1
ALPHA_TAIL  = 1.0
TAIL_CLIP   = 6.0

# -----------------------------
# Latent CycleGAN / WGAN-GP hyperparameters
# Preserved from the uploaded notebook.
# -----------------------------
GAN_BATCH_SIZE = 32
GAN_EPOCHS = 300
GAN_GP_WEIGHT = 10.0
GAN_DRIFT_EPS = 1e-3
GAN_K_PAIR = 5
GAN_BANK_SAMPLE = 512
GAN_LAMBDA_CYCLE = 5.0
GAN_LAMBDA_IDENTITY = 0.5
GAN_PAIR_W = 1.0
GAN_MMD_W = 1.0
GAN_COV_W = 0.2
GAN_MOM_W = 2.0
GAN_N_CRITIC = 5
GAN_VAL_EVERY = 10

# Saving translated train/val expression can be very large.
SAVE_TRANSLATED_TRAIN_VAL_EXPRESSION = False

# Run all conditions when True. Set False while debugging one condition.
RUN_ALL = False
DEBUG_CONDITION = "APOE"

# %%
# ============================================================
# 2. Preflight checks and common gene space
# ============================================================
def set_all_seeds(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_header_genes(path: str) -> List[str]:
    df0 = pd.read_csv(path, index_col=0, nrows=0)
    genes = [str(x) for x in df0.columns]
    if len(genes) != len(set(genes)):
        dup = pd.Index(genes)[pd.Index(genes).duplicated()].unique().tolist()
        raise ValueError(
            f"Duplicate gene columns in {path}. Example duplicates: {dup[:10]}"
        )
    return genes


def validate_input_paths():
    missing = []
    for cond in RUN_CONDITIONS:
        if cond not in INPUT_FILES:
            missing.append(f"{cond}: missing condition entry")
            continue
        for state in ["NPC", "Neuron"]:
            p = INPUT_FILES[cond].get(state)
            if not p or not Path(p).exists():
                missing.append(f"{cond} {state}: {p}")
    if missing:
        msg = "\n".join(["Missing input files:"] + [f"  - {x}" for x in missing])
        raise FileNotFoundError(msg)
    print("✓ All configured expression files exist.")


def build_common_gene_space() -> np.ndarray:
    """
    Uses headers only, so all 12 full matrices do not need to be held in RAM.
    Gene order follows control NPC, restricted to the intersection across all files.
    """
    header_genes = {}
    for cond in RUN_CONDITIONS:
        for state in ["NPC", "Neuron"]:
            key = f"{cond}_{state}"
            header_genes[key] = _read_header_genes(INPUT_FILES[cond][state])
            print(f"{key:20s}: {len(header_genes[key]):,} genes")

    common = set(header_genes["control_NPC"])
    for genes in header_genes.values():
        common &= set(genes)

    ordered = np.array(
        [g for g in header_genes["control_NPC"] if g in common],
        dtype=object,
    )

    if len(ordered) == 0:
        raise ValueError("No common genes found across the 12 configured files.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_ROOT / "gene_names.npy", ordered)
    pd.DataFrame({"gene": ordered}).to_csv(OUTPUT_ROOT / "gene_names.csv", index=False)

    print(f"\n✓ Common gene space: {len(ordered):,} genes")
    print(f"✓ Saved fixed gene order to: {OUTPUT_ROOT / 'gene_names.npy'}")
    return ordered


def load_expression_csv(path: str, genes: np.ndarray) -> pd.DataFrame:
    """Read cells × genes; preserve non-unique original barcodes as metadata.

    A barcode is not necessarily a globally unique cell ID (e.g., barcodes
    can be reused across libraries). Rows, NOT barcode labels, are the units
    of analysis, so every row receives a stable positional identifier.
    """
    df = pd.read_csv(path, index_col=0)
    barcodes = df.index.astype(str).to_numpy(copy=True)
    df.columns = df.columns.astype(str)

    # Row IDs are unique within each input file and independent of barcode reuse.
    df.index = pd.Index(
        [f"cell_{i:08d}" for i in range(len(df))], name="cell_id"
    )
    if not df.columns.is_unique:
        raise ValueError(f"Gene columns are not unique in {path}")

    missing = [g for g in genes if g not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} common genes unexpectedly.")

    df = df.loc[:, genes]
    df = df.apply(pd.to_numeric, errors="raise")

    X = df.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError(f"{path}: contains NaN/Inf values.")
    if X.size and np.min(X) < -1e-6:
        raise ValueError(
            f"{path}: contains negative values. HurdleAE expects non-negative expression."
        )

    # Metadata follows input row order; it is NOT used as an alignment key.
    df.attrs["barcodes"] = barcodes
    n_duplicates = len(barcodes) - len(set(barcodes))
    if n_duplicates:
        print(f"INFO: {Path(path).name}: {n_duplicates:,} repeated barcode rows; "
              "using unique row-based cell IDs and preserving original barcodes.")
    return df


def load_groups(group_path: Optional[str], cell_ids: pd.Index) -> Optional[np.ndarray]:
    if group_path is None:
        return None
    g = pd.read_csv(group_path, index_col=0)
    g.index = g.index.astype(str)
    if GROUP_COLUMN not in g.columns:
        raise ValueError(f"{group_path}: missing required column '{GROUP_COLUMN}'")
    missing = cell_ids.difference(g.index)
    if len(missing):
        raise ValueError(
            f"{group_path}: group labels missing for {len(missing)} expression cells."
        )
    return g.loc[cell_ids, GROUP_COLUMN].astype(str).to_numpy()


set_all_seeds(SEED)
validate_input_paths()
GENE_NAMES = build_common_gene_space()

# %%
# ============================================================
# 3. Train / validation / test splitting
# ============================================================
def make_split_indices(
    n: int,
    groups: Optional[np.ndarray] = None,
    seed: int = SEED,
) -> Dict[str, np.ndarray]:
    """
    Returns mutually exclusive train/val/test indices.

    - groups=None: cell-level random split.
    - groups supplied: entire groups are held out together.
    """
    idx = np.arange(n)

    if groups is None:
        train_idx, temp_idx = train_test_split(
            idx,
            train_size=TRAIN_FRAC,
            random_state=seed,
            shuffle=True,
        )
        test_ratio_within_temp = TEST_FRAC / (VAL_FRAC + TEST_FRAC)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_ratio_within_temp,
            random_state=seed + 1,
            shuffle=True,
        )
    else:
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError("groups length does not match number of cells.")
        if len(np.unique(groups)) < 3:
            raise ValueError(
                "Group-held-out splitting requires at least 3 distinct groups."
            )

        gss1 = GroupShuffleSplit(
            n_splits=1, train_size=TRAIN_FRAC, random_state=seed
        )
        train_idx, temp_idx = next(gss1.split(idx, groups=groups))

        temp_groups = groups[temp_idx]
        test_ratio_within_temp = TEST_FRAC / (VAL_FRAC + TEST_FRAC)
        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=test_ratio_within_temp,
            random_state=seed + 1,
        )
        val_rel, test_rel = next(
            gss2.split(temp_idx, groups=temp_groups)
        )
        val_idx = temp_idx[val_rel]
        test_idx = temp_idx[test_rel]

        # No group overlap across partitions.
        gt = set(groups[train_idx])
        gv = set(groups[val_idx])
        ge = set(groups[test_idx])
        assert gt.isdisjoint(gv)
        assert gt.isdisjoint(ge)
        assert gv.isdisjoint(ge)

    train_idx = np.sort(np.asarray(train_idx, dtype=int))
    val_idx   = np.sort(np.asarray(val_idx, dtype=int))
    test_idx  = np.sort(np.asarray(test_idx, dtype=int))

    assert len(np.intersect1d(train_idx, val_idx)) == 0
    assert len(np.intersect1d(train_idx, test_idx)) == 0
    assert len(np.intersect1d(val_idx, test_idx)) == 0
    assert len(np.union1d(np.union1d(train_idx, val_idx), test_idx)) == n

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_split_assignments(
    out_dir: Path,
    state: str,
    cell_ids: pd.Index,
    splits: Dict[str, np.ndarray],
    groups: Optional[np.ndarray] = None,
    barcodes: Optional[np.ndarray] = None,
):
    labels = np.empty(len(cell_ids), dtype=object)
    for split_name, split_idx in splits.items():
        labels[split_idx] = split_name

    out = pd.DataFrame({
        "cell_id": cell_ids.astype(str),
        "state": state,
        "split": labels,
    })
    if barcodes is not None:
        if len(barcodes) != len(out):
            raise ValueError("barcode metadata length differs from number of cells")
        out.insert(1, "barcode", np.asarray(barcodes).astype(str))
    if groups is not None:
        out["group"] = groups.astype(str)

    out.to_csv(out_dir / f"split_assignments_{state.lower()}.csv", index=False)


def print_split_summary(
    condition: str,
    state: str,
    splits: Dict[str, np.ndarray],
    groups: Optional[np.ndarray] = None,
):
    mode = "group-held-out" if groups is not None else "cell-level"
    print(f"{condition} {state} ({mode})")
    for k in ["train", "val", "test"]:
        extra = ""
        if groups is not None:
            extra = f" | groups={len(np.unique(groups[splits[k]]))}"
        print(f"  {k:5s}: {len(splits[k]):6d}{extra}")

# %%
# ============================================================
# 4. HurdleAE: separate NPC / Neuron encoders + shared decoder
# ============================================================
class NumpyExpressionDataset(Dataset):
    def __init__(self, X):
        X = np.asarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("Expression matrix contains NaN/Inf.")
        self.x = torch.from_numpy(X)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]


def _enc_block(d_in, d_out, dropout):
    return nn.Sequential(
        nn.Linear(d_in, d_out),
        nn.LayerNorm(d_out),
        nn.GELU(),
        nn.Dropout(dropout),
    )


def _dec_block(d_in, d_out):
    return nn.Sequential(
        nn.Linear(d_in, d_out),
        nn.LayerNorm(d_out),
        nn.GELU(),
    )


def _build_encoder(input_dim, latent_dim, dropout):
    return nn.Sequential(
        _enc_block(input_dim, 512, dropout),
        _enc_block(512, 256, dropout),
        nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        ),
        nn.Linear(128, latent_dim),
    )


class SharedHurdleDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            _dec_block(latent_dim, 128),
            _dec_block(128, 256),
            _dec_block(256, 512),
        )
        self.mask_logits = nn.Linear(512, output_dim)
        self.mag_raw = nn.Linear(512, output_dim)
        self.softplus = nn.Softplus()

    def forward(self, z):
        h = self.trunk(z)
        logits = self.mask_logits(h)
        p = torch.sigmoid(logits)
        mag = self.softplus(self.mag_raw(h))
        x_soft = p * mag
        return x_soft, logits, p, mag


class DevelopmentalHurdleAE(nn.Module):
    """
    source = NPC
    target = Neuron
    """
    def __init__(self, input_dim, latent_dim=128, dropout=0.10):
        super().__init__()
        self.encoder_source = _build_encoder(input_dim, latent_dim, dropout)
        self.encoder_target = _build_encoder(input_dim, latent_dim, dropout)
        self.decoder = SharedHurdleDecoder(latent_dim, input_dim)

    def encode(self, x, domain: str):
        if domain == "source":
            return self.encoder_source(x)
        if domain == "target":
            return self.encoder_target(x)
        raise ValueError("domain must be 'source' or 'target'")

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, domain: str):
        z = self.encode(x, domain)
        x_soft, logits, p, mag = self.decode(z)
        return z, x_soft, logits, p, mag


def corrupt_nonzero_entries(x, drop_prob=0.05):
    if drop_prob <= 0:
        return x
    nz = x > 0
    drop = (torch.rand_like(x) < drop_prob) & nz
    out = x.clone()
    out[drop] = 0.0
    return out


def compute_pos_weight_per_gene(X_train, thr=THR, clip_max=50.0):
    X = np.asarray(X_train, dtype=np.float32)
    pos = (X > thr).sum(axis=0).astype(np.float32)
    neg = X.shape[0] - pos
    pos = np.maximum(pos, 1.0)
    pw = np.clip(neg / pos, 1.0, clip_max).astype(np.float32)
    return torch.from_numpy(pw)


def masked_smooth_l1(
    pred,
    target,
    mask,
    beta=HUBER_BETA,
    alpha_tail=ALPHA_TAIL,
    tail_clip=TAIL_CLIP,
    eps=1e-8,
):
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    tail_w = 1.0 + alpha_tail * torch.clamp(target, max=tail_clip)
    loss = loss * tail_w
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / (denom + eps)


def hurdle_loss(
    logits,
    p,
    mag,
    x_true,
    pos_weight_gene,
    thr=THR,
):
    mask = (x_true > thr).float()
    zmask = 1.0 - mask

    bce = F.binary_cross_entropy_with_logits(
        logits,
        mask,
        pos_weight=pos_weight_gene.view(1, -1),
        reduction="mean",
    )

    mag_nz = masked_smooth_l1(
        mag,
        x_true,
        mask,
        beta=HUBER_BETA,
        alpha_tail=ALPHA_TAIL,
        tail_clip=TAIL_CLIP,
    )

    fp = (p * zmask).sum() / zmask.sum().clamp_min(1.0)
    x_soft = p * mag
    soft = F.smooth_l1_loss(
        x_soft, x_true, reduction="mean", beta=HUBER_BETA
    )

    return (
        LAMBDA_BCE * bce
        + LAMBDA_MAG * mag_nz
        + LAMBDA_FP * fp
        + LAMBDA_SOFT * soft
    )


def _infinite_batches(loader):
    while True:
        for batch in loader:
            yield batch


def _eval_ae_domain(model, loader, device, domain, pos_weight):
    total_loss = 0.0
    total_n = 0
    zvars = []

    model.eval()
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            z, _, logits, p, mag = model(x, domain)
            loss = hurdle_loss(logits, p, mag, x, pos_weight)
            n = x.shape[0]
            total_loss += float(loss.item()) * n
            total_n += n
            zvars.append(float(z.var(dim=0).mean().item()))

    return (
        total_loss / max(total_n, 1),
        float(np.mean(zvars)) if zvars else np.nan,
    )


def train_hurdle_ae(
    X_source,
    X_target,
    source_splits,
    target_splits,
    out_dir: Path,
):
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_loader(X, idx, shuffle):
        return DataLoader(
            NumpyExpressionDataset(X[idx]),
            batch_size=AE_BATCH_SIZE,
            shuffle=shuffle,
            drop_last=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    src_train_dl = make_loader(X_source, source_splits["train"], True)
    tgt_train_dl = make_loader(X_target, target_splits["train"], True)
    src_val_dl   = make_loader(X_source, source_splits["val"], False)
    tgt_val_dl   = make_loader(X_target, target_splits["val"], False)

    model = DevelopmentalHurdleAE(
        input_dim=X_source.shape[1],
        latent_dim=LATENT_DIM,
        dropout=AE_DROPOUT,
    ).to(device)

    pos_w_source = compute_pos_weight_per_gene(
        X_source[source_splits["train"]]
    ).to(device)
    pos_w_target = compute_pos_weight_per_gene(
        X_target[target_splits["train"]]
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=AE_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5
    )

    src_iter = _infinite_batches(src_train_dl)
    tgt_iter = _infinite_batches(tgt_train_dl)
    steps_per_epoch = max(len(src_train_dl), len(tgt_train_dl))

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(1, AE_EPOCHS + 1):
        model.train()
        running = 0.0

        for _ in range(steps_per_epoch):
            xs = next(src_iter).to(device, non_blocking=True)
            xt = next(tgt_iter).to(device, non_blocking=True)

            xs_in = corrupt_nonzero_entries(xs, INPUT_DROP_PROB)
            xt_in = corrupt_nonzero_entries(xt, INPUT_DROP_PROB)

            _, _, ls, ps, ms = model(xs_in, "source")
            _, _, lt, pt, mt = model(xt_in, "target")

            loss_s = hurdle_loss(ls, ps, ms, xs, pos_w_source)
            loss_t = hurdle_loss(lt, pt, mt, xt, pos_w_target)

            # Equal domain weighting, independent of cell counts.
            loss = 0.5 * (loss_s + loss_t)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += float(loss.item())

        train_loss = running / max(steps_per_epoch, 1)

        val_source, zvar_source = _eval_ae_domain(
            model, src_val_dl, device, "source", pos_w_source
        )
        val_target, zvar_target = _eval_ae_domain(
            model, tgt_val_dl, device, "target", pos_w_target
        )
        val_loss = 0.5 * (val_source + val_target)
        sched.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_source_loss": val_source,
            "val_target_loss": val_target,
            "val_source_latent_var": zvar_source,
            "val_target_latent_var": zvar_target,
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(row)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"AE epoch {epoch:03d} | train={train_loss:.4f} "
                f"| val={val_loss:.4f} | lr={row['lr']:.2e}"
            )

        if val_loss < best_val - AE_MIN_DELTA:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= AE_PATIENCE:
                print(
                    f"AE early stop at epoch {epoch}; "
                    f"best epoch={best_epoch}, val={best_val:.4f}"
                )
                break

    if best_state is None:
        raise RuntimeError("AE training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "ae_training_history.csv", index=False)

    ckpt_path = out_dir / "hurdle_ae_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": int(X_source.shape[1]),
                "latent_dim": LATENT_DIM,
                "dropout": AE_DROPOUT,
                "best_epoch": int(best_epoch),
                "best_val": float(best_val),
            },
            "source_splits": source_splits,
            "target_splits": target_splits,
        },
        ckpt_path,
    )

    return model, device, history_df


def infer_z_p_mag(model, X, device, domain, batch_size=256):
    dl = DataLoader(
        NumpyExpressionDataset(X),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    Zs, Ps, MAGs = [], [], []
    model.eval()
    with torch.no_grad():
        for x in dl:
            x = x.to(device)
            z, _, _, p, mag = model(x, domain)
            Zs.append(z.cpu().numpy())
            Ps.append(p.cpu().numpy())
            MAGs.append(mag.cpu().numpy())

    return (
        np.concatenate(Zs).astype(np.float32),
        np.concatenate(Ps).astype(np.float32),
        np.concatenate(MAGs).astype(np.float32),
    )


def calibrate_tau_gene(P_train, X_train, thr=THR):
    p0_real = (np.asarray(X_train, np.float32) <= thr).mean(axis=0)
    tau = np.array(
        [
            np.quantile(P_train[:, g], float(p0_real[g]))
            for g in range(P_train.shape[1])
        ],
        dtype=np.float32,
    )
    return tau


def z_standardize_stats(Z_train, eps=1e-6):
    mean = Z_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(
        Z_train.std(axis=0, keepdims=True), eps
    ).astype(np.float32)
    return mean, std


def standardize(Z, mean, std):
    return ((Z - mean) / std).astype(np.float32)


def unstandardize(Z_std, mean, std):
    return (Z_std * std + mean).astype(np.float32)

# %%
# ============================================================
# 5. Latent CycleGAN / WGAN-GP: NPC latent → Neuron latent
# ============================================================
def generator_loss(fake_scores):
    return -tf.reduce_mean(fake_scores)


def discriminator_loss(real_scores, fake_scores):
    return tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)


def gradient_penalty(discriminator, real, fake):
    alpha = tf.random.uniform([tf.shape(real)[0], 1])
    inter = real + alpha * (fake - real)
    with tf.GradientTape() as tape:
        tape.watch(inter)
        pred = discriminator(inter)
    grad = tape.gradient(pred, inter)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1) + 1e-12)
    return tf.reduce_mean((norm - 1.0) ** 2)


def cov_loss(fake, real):
    def cov(x):
        xc = x - tf.reduce_mean(x, axis=0, keepdims=True)
        n = tf.cast(tf.maximum(tf.shape(xc)[0] - 1, 1), tf.float32)
        return tf.matmul(xc, xc, transpose_a=True) / n

    return tf.reduce_mean(tf.square(cov(fake) - cov(real)))


def moment_loss(fake, real, eps=1e-6):
    mf = tf.reduce_mean(fake, axis=0)
    mr = tf.reduce_mean(real, axis=0)
    sf = tf.math.reduce_std(fake, axis=0) + eps
    sr = tf.math.reduce_std(real, axis=0) + eps
    return tf.reduce_mean(tf.abs(mf - mr)) + tf.reduce_mean(tf.abs(sf - sr))


def mmd_rbf(x, y, sigmas):
    def kernel(a, b, s):
        gamma = 1.0 / (2.0 * s * s)
        a2 = tf.reduce_sum(a * a, axis=1, keepdims=True)
        b2 = tf.reduce_sum(b * b, axis=1, keepdims=True)
        d2 = a2 - 2.0 * tf.matmul(a, b, transpose_b=True) + tf.transpose(b2)
        return tf.exp(-gamma * tf.maximum(d2, 0.0))

    kxx = tf.add_n([kernel(x, x, s) for s in sigmas]) / len(sigmas)
    kyy = tf.add_n([kernel(y, y, s) for s in sigmas]) / len(sigmas)
    kxy = tf.add_n([kernel(x, y, s) for s in sigmas]) / len(sigmas)
    return tf.reduce_mean(kxx) + tf.reduce_mean(kyy) - 2.0 * tf.reduce_mean(kxy)


def choose_mmd_sigmas(Z_source_train, Z_target_train, seed=SEED):
    """
    Data-adaptive MMD bandwidths estimated from TRAINING latents only.
    This replaces the hard-coded APOE-specific distance assumption.
    """
    rng = np.random.default_rng(seed)
    ns = min(256, len(Z_source_train))
    nt = min(256, len(Z_target_train))
    s = Z_source_train[rng.choice(len(Z_source_train), ns, replace=False)]
    t = Z_target_train[rng.choice(len(Z_target_train), nt, replace=False)]

    d = np.linalg.norm(s[:, None, :] - t[None, :, :], axis=-1)
    med = float(np.median(d))
    med = max(med, 1e-3)
    sigmas = tuple(float(med * f) for f in (0.25, 0.5, 1.0, 2.0))
    print(f"Training median cross-domain latent distance: {med:.3f}")
    print(f"MMD sigmas: {sigmas}")
    return sigmas


def knn_pair_loss_cos(fake_target, target_bank, k=5, bank_sample=512):
    n_bank = tf.shape(target_bank)[0]
    m = tf.minimum(bank_sample, n_bank)
    idx = tf.random.uniform([m], 0, n_bank, dtype=tf.int32)
    bank = tf.gather(target_bank, idx)

    f = tf.math.l2_normalize(fake_target, axis=1)
    b = tf.math.l2_normalize(bank, axis=1)
    sims = tf.matmul(f, b, transpose_b=True)

    k_eff = tf.minimum(tf.cast(k, tf.int32), m)
    _, inds = tf.math.top_k(sims, k=k_eff)
    b_knn = tf.gather(b, inds, axis=0)
    b_tgt = tf.reduce_mean(b_knn, axis=1)

    return tf.reduce_mean(
        1.0 - tf.reduce_sum(f * b_tgt, axis=1)
    )


def build_generator(dim, name):
    """
    Symmetric architecture in both directions:
    d -> 1024 -> 512 -> 512 -> d
    """
    x_in = Input(shape=(dim,))
    x = Dense(1024, activation="relu")(x_in)
    x = LayerNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = LayerNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = LayerNormalization()(x)
    out = Dense(dim)(x)
    return models.Model(x_in, out, name=name)


def build_critic(dim, name):
    x_in = Input(shape=(dim,))
    x = Dense(512, activation="relu")(x_in)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    out = Dense(1)(x)
    return models.Model(x_in, out, name=name)


def gan_validation_score(
    G_ST,
    G_TS,
    Z_source_val,
    Z_target_val,
    sigmas,
    max_cells=512,
):
    """
    Monitoring only. Lower is better.
    Test data are never used here.
    """
    ns = min(max_cells, len(Z_source_val))
    nt = min(max_cells, len(Z_target_val))
    src = Z_source_val[:ns].astype(np.float32)
    tgt = Z_target_val[:nt].astype(np.float32)

    pred = G_ST(tf.constant(src), training=False)
    cycled = G_TS(pred, training=False)

    mmd = float(
        mmd_rbf(pred, tf.constant(tgt), sigmas).numpy()
    )
    mean_gap = float(
        np.mean(np.abs(pred.numpy().mean(0) - tgt.mean(0)))
    )
    std_gap = float(
        np.mean(np.abs(pred.numpy().std(0) - tgt.std(0)))
    )
    cycle_gap = float(
        np.mean(np.abs(cycled.numpy() - src))
    )
    score = mmd + mean_gap + std_gap + cycle_gap

    return {
        "val_score": score,
        "val_mmd": mmd,
        "val_mean_gap": mean_gap,
        "val_std_gap": std_gap,
        "val_cycle_l1": cycle_gap,
    }


def train_latent_cyclegan(
    Z_source_train,
    Z_target_train,
    Z_source_val,
    Z_target_val,
    out_dir: Path,
    condition: str,
):
    set_all_seeds(SEED)

    dim = Z_source_train.shape[1]
    assert dim == Z_target_train.shape[1]

    sigmas = choose_mmd_sigmas(
        Z_source_train, Z_target_train, seed=SEED
    )

    source_bank = Z_source_train.astype(np.float32)
    target_bank_np = Z_target_train.astype(np.float32)
    target_bank = tf.constant(target_bank_np, dtype=tf.float32)

    shuffle_source = max(1, min(len(source_bank), 4096))
    shuffle_target = max(1, min(len(target_bank_np), 4096))

    ds_source = (
        tf.data.Dataset.from_tensor_slices(source_bank)
        .shuffle(shuffle_source, seed=SEED, reshuffle_each_iteration=True)
        .repeat()
    )
    ds_target = (
        tf.data.Dataset.from_tensor_slices(target_bank_np)
        .shuffle(shuffle_target, seed=SEED + 1, reshuffle_each_iteration=True)
        .repeat()
    )
    train_dataset = (
        tf.data.Dataset.zip((ds_source, ds_target))
        .batch(GAN_BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    steps_per_epoch = max(
        math.ceil(len(source_bank) / GAN_BATCH_SIZE),
        math.ceil(len(target_bank_np) / GAN_BATCH_SIZE),
    )

    G_ST = build_generator(dim, "G_NPC_to_Neuron")
    G_TS = build_generator(dim, "G_Neuron_to_NPC")
    D_T = build_critic(dim, "D_Neuron")
    D_S = build_critic(dim, "D_NPC")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True,
    )

    g_st_opt = Adam(learning_rate=lr_schedule, beta_1=0.5, clipnorm=1.0)
    g_ts_opt = Adam(learning_rate=lr_schedule, beta_1=0.5, clipnorm=1.0)
    d_t_opt  = Adam(learning_rate=lr_schedule, beta_1=0.5, clipnorm=1.0)
    d_s_opt  = Adam(learning_rate=lr_schedule, beta_1=0.5, clipnorm=1.0)

    @tf.function
    def train_step(real_source, real_target):
        for _ in tf.range(GAN_N_CRITIC):
            with tf.GradientTape(persistent=True) as tape_d:
                fake_target = G_ST(real_source, training=True)
                fake_source = G_TS(real_target, training=True)

                rt = D_T(real_target, training=True)
                ft = D_T(fake_target, training=True)
                rs = D_S(real_source, training=True)
                fs = D_S(fake_source, training=True)

                gp_t = gradient_penalty(D_T, real_target, fake_target)
                gp_s = gradient_penalty(D_S, real_source, fake_source)

                drift_t = GAN_DRIFT_EPS * (
                    tf.reduce_mean(tf.square(rt))
                    + tf.reduce_mean(tf.square(ft))
                )
                drift_s = GAN_DRIFT_EPS * (
                    tf.reduce_mean(tf.square(rs))
                    + tf.reduce_mean(tf.square(fs))
                )

                d_t_loss = (
                    discriminator_loss(rt, ft)
                    + GAN_GP_WEIGHT * gp_t
                    + drift_t
                )
                d_s_loss = (
                    discriminator_loss(rs, fs)
                    + GAN_GP_WEIGHT * gp_s
                    + drift_s
                )

            d_t_grads = tape_d.gradient(
                d_t_loss, D_T.trainable_variables
            )
            d_s_grads = tape_d.gradient(
                d_s_loss, D_S.trainable_variables
            )
            d_t_opt.apply_gradients(zip(d_t_grads, D_T.trainable_variables))
            d_s_opt.apply_gradients(zip(d_s_grads, D_S.trainable_variables))
            del tape_d

        wasserstein_gap = tf.reduce_mean(rt) - tf.reduce_mean(ft)

        with tf.GradientTape(persistent=True) as tape_g:
            fake_target = G_ST(real_source, training=True)
            fake_source = G_TS(real_target, training=True)

            adv_st = generator_loss(D_T(fake_target, training=True))
            adv_ts = generator_loss(D_S(fake_source, training=True))

            cycled_source = G_TS(fake_target, training=True)
            cycled_target = G_ST(fake_source, training=True)

            cyc_source = tf.reduce_mean(
                tf.abs(real_source - cycled_source)
            )
            cyc_target = tf.reduce_mean(
                tf.abs(real_target - cycled_target)
            )
            cycle_total = cyc_source + cyc_target

            id_target = tf.reduce_mean(
                tf.abs(real_target - G_ST(real_target, training=True))
            )
            id_source = tf.reduce_mean(
                tf.abs(real_source - G_TS(real_source, training=True))
            )

            pair = knn_pair_loss_cos(
                fake_target,
                target_bank,
                k=GAN_K_PAIR,
                bank_sample=GAN_BANK_SAMPLE,
            )
            mmd = mmd_rbf(fake_target, real_target, sigmas)
            cov = cov_loss(fake_target, real_target)
            mom_st = moment_loss(fake_target, real_target)
            mom_ts = moment_loss(fake_source, real_source)

            g_st_loss = (
                adv_st
                + GAN_LAMBDA_CYCLE * cycle_total
                + GAN_LAMBDA_IDENTITY * id_target
                + GAN_PAIR_W * pair
                + GAN_MMD_W * mmd
                + GAN_COV_W * cov
                + GAN_MOM_W * mom_st
            )
            g_ts_loss = (
                adv_ts
                + GAN_LAMBDA_CYCLE * cycle_total
                + GAN_LAMBDA_IDENTITY * id_source
                + GAN_MOM_W * mom_ts
            )

        g_st_grads = tape_g.gradient(
            g_st_loss, G_ST.trainable_variables
        )
        g_ts_grads = tape_g.gradient(
            g_ts_loss, G_TS.trainable_variables
        )
        g_st_opt.apply_gradients(zip(g_st_grads, G_ST.trainable_variables))
        g_ts_opt.apply_gradients(zip(g_ts_grads, G_TS.trainable_variables))
        del tape_g

        return {
            "g_st_loss": g_st_loss,
            "g_ts_loss": g_ts_loss,
            "d_target_loss": d_t_loss,
            "d_source_loss": d_s_loss,
            "cycle_loss": cycle_total,
            "identity_loss": id_target + id_source,
            "pair_loss": pair,
            "mmd_loss": mmd,
            "cov_loss": cov,
            "moment_st_loss": mom_st,
            "moment_ts_loss": mom_ts,
            "wasserstein_gap": wasserstein_gap,
        }

    hist = []
    iterator = iter(train_dataset)

    for epoch in range(1, GAN_EPOCHS + 1):
        accum = {}
        for _ in range(steps_per_epoch):
            real_s, real_t = next(iterator)
            out = train_step(real_s, real_t)
            for k, v in out.items():
                accum[k] = accum.get(k, 0.0) + float(v.numpy())

        row = {
            "epoch": epoch,
            **{
                k: v / steps_per_epoch
                for k, v in accum.items()
            },
        }

        if epoch == 1 or epoch % GAN_VAL_EVERY == 0:
            row.update(
                gan_validation_score(
                    G_ST,
                    G_TS,
                    Z_source_val,
                    Z_target_val,
                    sigmas,
                )
            )
            print(
                f"GAN {condition} epoch {epoch:03d} | "
                f"G={row['g_st_loss']:.3f} | "
                f"D={row['d_target_loss']:.3f} | "
                f"MMD={row['mmd_loss']:.3f} | "
                f"val={row.get('val_score', np.nan):.3f}"
            )

        hist.append(row)

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(out_dir / "gan_training_history.csv", index=False)

    G_ST.save(out_dir / "G_NPC_to_Neuron.keras")
    G_TS.save(out_dir / "G_Neuron_to_NPC.keras")

    with open(out_dir / "gan_mmd_sigmas.json", "w") as f:
        json.dump({"sigmas": list(sigmas)}, f, indent=2)

    return G_ST, G_TS, hist_df, sigmas

# %%
# ============================================================
# 6. Decode translated Neuron latents and evaluate held-out test cells
# ============================================================
def decode_target_latents(
    ae_model,
    Z_target_raw,
    tau_target,
    device,
    batch_size=256,
):
    ds = NumpyExpressionDataset(Z_target_raw)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    X_soft, P_all, MAG_all = [], [], []

    ae_model.eval()
    with torch.no_grad():
        for z in dl:
            z = z.to(device)
            x_soft, _, p, mag = ae_model.decode(z)
            X_soft.append(x_soft.cpu().numpy())
            P_all.append(p.cpu().numpy())
            MAG_all.append(mag.cpu().numpy())

    X_soft = np.concatenate(X_soft).astype(np.float32)
    P = np.concatenate(P_all).astype(np.float32)
    MAG = np.concatenate(MAG_all).astype(np.float32)

    gate = (P > tau_target.reshape(1, -1)).astype(np.float32)
    X_hard = (gate * MAG).astype(np.float32)

    return X_soft, X_hard, P, MAG


def safe_pearson(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return pearsonr(x, y)


def safe_spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    return spearmanr(x, y)


def evaluate_translation(
    condition: str,
    X_source_test,
    X_target_test,
    X_pred_test,
    Z_target_test_std,
    Z_pred_test_std,
    gene_names,
    sigmas,
):
    """
    Population-level evaluation for unpaired source/target cells.
    """
    source_mean = X_source_test.mean(axis=0)
    target_mean = X_target_test.mean(axis=0)
    pred_mean = X_pred_test.mean(axis=0)

    target_det = (X_target_test > THR).mean(axis=0)
    pred_det = (X_pred_test > THR).mean(axis=0)

    true_delta = target_mean - source_mean
    pred_delta = pred_mean - source_mean

    mean_r, mean_p = safe_pearson(target_mean, pred_mean)
    mean_rho, mean_rho_p = safe_spearman(target_mean, pred_mean)

    det_r, det_p = safe_pearson(target_det, pred_det)
    det_rho, det_rho_p = safe_spearman(target_det, pred_det)

    delta_r, delta_p = safe_pearson(true_delta, pred_delta)
    delta_rho, delta_rho_p = safe_spearman(true_delta, pred_delta)

    # Top developmental-change recovery (absolute mean change ranking).
    def top_change_jaccard(k):
        k = min(int(k), len(gene_names))
        true_top = set(np.argpartition(np.abs(true_delta), -k)[-k:])
        pred_top = set(np.argpartition(np.abs(pred_delta), -k)[-k:])
        return len(true_top & pred_top) / max(len(true_top | pred_top), 1)

    top100_jaccard = top_change_jaccard(100)
    top200_jaccard = top_change_jaccard(200)
    top500_jaccard = top_change_jaccard(500)

    k_dir = min(500, len(gene_names))
    true_top500_idx = np.argpartition(np.abs(true_delta), -k_dir)[-k_dir:]
    top500_sign_concordance = float(
        np.mean(
            np.sign(true_delta[true_top500_idx])
            == np.sign(pred_delta[true_top500_idx])
        )
    )

    ks_stats = np.empty(len(gene_names), dtype=np.float32)
    ks_pvals = np.empty(len(gene_names), dtype=np.float64)
    wdist = np.empty(len(gene_names), dtype=np.float32)

    for g in range(len(gene_names)):
        ks = ks_2samp(X_target_test[:, g], X_pred_test[:, g])
        ks_stats[g] = ks.statistic
        ks_pvals[g] = ks.pvalue
        wdist[g] = wasserstein_distance(
            X_target_test[:, g],
            X_pred_test[:, g],
        )

    _, ks_fdr, _, _ = multipletests(ks_pvals, method="fdr_bh")

    per_gene = pd.DataFrame({
        "gene": gene_names,
        "source_test_mean": source_mean,
        "real_neuron_test_mean": target_mean,
        "translated_neuron_test_mean": pred_mean,
        "true_NPC_to_Neuron_delta": true_delta,
        "predicted_NPC_to_Neuron_delta": pred_delta,
        "real_neuron_detection_rate": target_det,
        "translated_neuron_detection_rate": pred_det,
        "ks_stat": ks_stats,
        "ks_pvalue": ks_pvals,
        "ks_fdr": ks_fdr,
        "wasserstein": wdist,
    })

    # Latent MMD on bounded test subsets for memory safety.
    n1 = min(512, len(Z_target_test_std))
    n2 = min(512, len(Z_pred_test_std))
    latent_mmd = float(
        mmd_rbf(
            tf.constant(Z_pred_test_std[:n2], dtype=tf.float32),
            tf.constant(Z_target_test_std[:n1], dtype=tf.float32),
            sigmas,
        ).numpy()
    )

    latent_mean_r, latent_mean_p = safe_pearson(
        Z_target_test_std.mean(axis=0),
        Z_pred_test_std.mean(axis=0),
    )

    summary = {
        "condition": condition,
        "n_source_test": int(X_source_test.shape[0]),
        "n_target_test": int(X_target_test.shape[0]),
        "n_genes": int(X_target_test.shape[1]),
        "gene_mean_pearson_r": float(mean_r),
        "gene_mean_pearson_p": float(mean_p),
        "gene_mean_spearman_rho": float(mean_rho),
        "gene_mean_spearman_p": float(mean_rho_p),
        "detection_pearson_r": float(det_r),
        "detection_pearson_p": float(det_p),
        "detection_spearman_rho": float(det_rho),
        "detection_spearman_p": float(det_rho_p),
        "developmental_delta_pearson_r": float(delta_r),
        "developmental_delta_pearson_p": float(delta_p),
        "developmental_delta_spearman_rho": float(delta_rho),
        "developmental_delta_spearman_p": float(delta_rho_p),
        "top100_change_jaccard": float(top100_jaccard),
        "top200_change_jaccard": float(top200_jaccard),
        "top500_change_jaccard": float(top500_jaccard),
        "top500_change_sign_concordance": float(top500_sign_concordance),
        "gene_mean_mae": float(np.mean(np.abs(target_mean - pred_mean))),
        "gene_mean_rmse": float(np.sqrt(np.mean((target_mean - pred_mean) ** 2))),
        "median_ks": float(np.median(ks_stats)),
        "mean_ks": float(np.mean(ks_stats)),
        "pct_genes_ks_lt_0_1": float(100.0 * np.mean(ks_stats < 0.1)),
        "median_wasserstein": float(np.median(wdist)),
        "mean_wasserstein": float(np.mean(wdist)),
        "real_target_zero_fraction": float(np.mean(X_target_test <= THR)),
        "translated_zero_fraction": float(np.mean(X_pred_test <= THR)),
        "latent_mean_pearson_r": float(latent_mean_r),
        "latent_mean_pearson_p": float(latent_mean_p),
        "latent_mmd": latent_mmd,
    }

    return summary, per_gene

# %%
# ============================================================
# 7. End-to-end runner for one condition
# ============================================================
def run_condition(condition: str) -> Dict[str, Any]:
    if condition not in INPUT_FILES:
        raise KeyError(f"Unknown condition: {condition}")

    print("\n" + "=" * 80)
    print(f"{condition}: NPC → Neuron")
    print("=" * 80)

    set_all_seeds(SEED)

    out_dir = OUTPUT_ROOT / f"{condition}_NPC_to_Neuron"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load only this pair, already aligned to global common genes
    # --------------------------------------------------------
    npc_df = load_expression_csv(
        INPUT_FILES[condition]["NPC"],
        GENE_NAMES,
    )
    neuron_df = load_expression_csv(
        INPUT_FILES[condition]["Neuron"],
        GENE_NAMES,
    )

    X_npc = npc_df.to_numpy(dtype=np.float32, copy=False)
    X_neuron = neuron_df.to_numpy(dtype=np.float32, copy=False)

    print(f"NPC:    {X_npc.shape}")
    print(f"Neuron: {X_neuron.shape}")

    npc_groups = load_groups(
        GROUP_FILES[condition]["NPC"],
        npc_df.index,
    )
    neuron_groups = load_groups(
        GROUP_FILES[condition]["Neuron"],
        neuron_df.index,
    )

    npc_splits = make_split_indices(
        len(npc_df),
        groups=npc_groups,
        seed=SEED,
    )
    neuron_splits = make_split_indices(
        len(neuron_df),
        groups=neuron_groups,
        seed=SEED + 100,
    )

    print_split_summary(
        condition, "NPC", npc_splits, npc_groups
    )
    print_split_summary(
        condition, "Neuron", neuron_splits, neuron_groups
    )

    save_split_assignments(
        out_dir,
        "NPC",
        npc_df.index,
        npc_splits,
        npc_groups,
        barcodes=npc_df.attrs["barcodes"],
    )
    save_split_assignments(
        out_dir,
        "Neuron",
        neuron_df.index,
        neuron_splits,
        neuron_groups,
        barcodes=neuron_df.attrs["barcodes"],
    )

    # --------------------------------------------------------
    # HurdleAE
    # --------------------------------------------------------
    ae_model, device, ae_history = train_hurdle_ae(
        X_npc,
        X_neuron,
        npc_splits,
        neuron_splits,
        out_dir,
    )

    Z_npc, P_npc, MAG_npc = infer_z_p_mag(
        ae_model, X_npc, device, "source"
    )
    Z_neuron, P_neuron, MAG_neuron = infer_z_p_mag(
        ae_model, X_neuron, device, "target"
    )

    # Fit latent scaling on TRAIN ONLY.
    mean_npc, std_npc = z_standardize_stats(
        Z_npc[npc_splits["train"]]
    )
    mean_neuron, std_neuron = z_standardize_stats(
        Z_neuron[neuron_splits["train"]]
    )

    # Fit hurdle thresholds on TRAIN ONLY.
    tau_npc = calibrate_tau_gene(
        P_npc[npc_splits["train"]],
        X_npc[npc_splits["train"]],
    )
    tau_neuron = calibrate_tau_gene(
        P_neuron[neuron_splits["train"]],
        X_neuron[neuron_splits["train"]],
    )

    np.save(out_dir / "tau_npc.npy", tau_npc)
    np.save(out_dir / "tau_neuron.npy", tau_neuron)

    np.savez_compressed(
        out_dir / "latent_spaces.npz",
        Z_npc_train=Z_npc[npc_splits["train"]],
        Z_npc_val=Z_npc[npc_splits["val"]],
        Z_npc_test=Z_npc[npc_splits["test"]],
        Z_neuron_train=Z_neuron[neuron_splits["train"]],
        Z_neuron_val=Z_neuron[neuron_splits["val"]],
        Z_neuron_test=Z_neuron[neuron_splits["test"]],
        mean_npc=mean_npc,
        std_npc=std_npc,
        mean_neuron=mean_neuron,
        std_neuron=std_neuron,
        npc_train_idx=npc_splits["train"],
        npc_val_idx=npc_splits["val"],
        npc_test_idx=npc_splits["test"],
        neuron_train_idx=neuron_splits["train"],
        neuron_val_idx=neuron_splits["val"],
        neuron_test_idx=neuron_splits["test"],
    )

    # --------------------------------------------------------
    # Standardized latent split arrays
    # --------------------------------------------------------
    Z_npc_train_std = standardize(
        Z_npc[npc_splits["train"]], mean_npc, std_npc
    )
    Z_npc_val_std = standardize(
        Z_npc[npc_splits["val"]], mean_npc, std_npc
    )
    Z_npc_test_std = standardize(
        Z_npc[npc_splits["test"]], mean_npc, std_npc
    )

    Z_neuron_train_std = standardize(
        Z_neuron[neuron_splits["train"]],
        mean_neuron,
        std_neuron,
    )
    Z_neuron_val_std = standardize(
        Z_neuron[neuron_splits["val"]],
        mean_neuron,
        std_neuron,
    )
    Z_neuron_test_std = standardize(
        Z_neuron[neuron_splits["test"]],
        mean_neuron,
        std_neuron,
    )

    # --------------------------------------------------------
    # Latent CycleGAN: NPC -> Neuron
    # --------------------------------------------------------
    G_ST, G_TS, gan_history, sigmas = train_latent_cyclegan(
        Z_npc_train_std,
        Z_neuron_train_std,
        Z_npc_val_std,
        Z_neuron_val_std,
        out_dir,
        condition,
    )

    # Translate each source split.
    Z_pred_train_std = G_ST.predict(
        Z_npc_train_std,
        batch_size=256,
        verbose=0,
    ).astype(np.float32)
    Z_pred_val_std = G_ST.predict(
        Z_npc_val_std,
        batch_size=256,
        verbose=0,
    ).astype(np.float32)
    Z_pred_test_std = G_ST.predict(
        Z_npc_test_std,
        batch_size=256,
        verbose=0,
    ).astype(np.float32)

    # Generator outputs are in target-standardized latent coordinates.
    Z_pred_train_raw = unstandardize(
        Z_pred_train_std, mean_neuron, std_neuron
    )
    Z_pred_val_raw = unstandardize(
        Z_pred_val_std, mean_neuron, std_neuron
    )
    Z_pred_test_raw = unstandardize(
        Z_pred_test_std, mean_neuron, std_neuron
    )

    np.savez_compressed(
        out_dir / "translated_latent_spaces.npz",
        Z_pred_neuron_train_std=Z_pred_train_std,
        Z_pred_neuron_val_std=Z_pred_val_std,
        Z_pred_neuron_test_std=Z_pred_test_std,
        Z_pred_neuron_train_raw=Z_pred_train_raw,
        Z_pred_neuron_val_raw=Z_pred_val_raw,
        Z_pred_neuron_test_raw=Z_pred_test_raw,
        Z_real_neuron_train_std=Z_neuron_train_std,
        Z_real_neuron_val_std=Z_neuron_val_std,
        Z_real_neuron_test_std=Z_neuron_test_std,
        Z_source_npc_train_std=Z_npc_train_std,
        Z_source_npc_val_std=Z_npc_val_std,
        Z_source_npc_test_std=Z_npc_test_std,
    )

    # --------------------------------------------------------
    # Decode held-out translated Neuron test latents
    # --------------------------------------------------------
    X_pred_soft_test, X_pred_hard_test, P_pred_test, MAG_pred_test = (
        decode_target_latents(
            ae_model,
            Z_pred_test_raw,
            tau_neuron,
            device,
        )
    )

    np.savez_compressed(
        out_dir / "translated_expression_test.npz",
        X_source_npc_test=X_npc[npc_splits["test"]],
        X_real_neuron_test=X_neuron[neuron_splits["test"]],
        X_translated_neuron_soft_test=X_pred_soft_test,
        X_translated_neuron_hard_test=X_pred_hard_test,
        P_translated_test=P_pred_test,
        MAG_translated_test=MAG_pred_test,
        npc_test_idx=npc_splits["test"],
        neuron_test_idx=neuron_splits["test"],
    )

    if SAVE_TRANSLATED_TRAIN_VAL_EXPRESSION:
        for split_name, Zraw in [
            ("train", Z_pred_train_raw),
            ("val", Z_pred_val_raw),
        ]:
            xsoft, xhard, _, _ = decode_target_latents(
                ae_model, Zraw, tau_neuron, device
            )
            np.savez_compressed(
                out_dir / f"translated_expression_{split_name}.npz",
                X_translated_neuron_soft=xsoft,
                X_translated_neuron_hard=xhard,
            )

    # --------------------------------------------------------
    # FINAL TEST METRICS ONLY
    # --------------------------------------------------------
    summary, per_gene = evaluate_translation(
        condition=condition,
        X_source_test=X_npc[npc_splits["test"]],
        X_target_test=X_neuron[neuron_splits["test"]],
        X_pred_test=X_pred_hard_test,
        Z_target_test_std=Z_neuron_test_std,
        Z_pred_test_std=Z_pred_test_std,
        gene_names=GENE_NAMES,
        sigmas=sigmas,
    )

    pd.DataFrame([summary]).to_csv(
        out_dir / "metrics_summary.csv",
        index=False,
    )
    per_gene.to_csv(
        out_dir / "metrics_per_gene.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Reproducibility manifest
    # --------------------------------------------------------
    manifest = {
        "condition": condition,
        "translation": "NPC_to_Neuron",
        "npc_input": INPUT_FILES[condition]["NPC"],
        "neuron_input": INPUT_FILES[condition]["Neuron"],
        "output_dir": str(out_dir),
        "n_genes": int(len(GENE_NAMES)),
        "split": {
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
            "test_frac": TEST_FRAC,
            "npc_split_mode": (
                "group-held-out" if npc_groups is not None else "cell-level"
            ),
            "neuron_split_mode": (
                "group-held-out" if neuron_groups is not None else "cell-level"
            ),
        },
        "seed": SEED,
        "latent_dim": LATENT_DIM,
        "ae_best_epoch": int(
            ae_history.loc[ae_history["val_loss"].idxmin(), "epoch"]
        ),
        "mmd_sigmas": list(sigmas),
        "evaluation_decode": "hard",
    }

    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nHeld-out test summary")
    print(pd.DataFrame([summary]).T.to_string(header=False))
    print(f"\n✓ Saved all outputs to: {out_dir}")

    # Free large objects before the next condition.
    del npc_df, neuron_df, X_npc, X_neuron
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tf.keras.backend.clear_session()

    return {
        "condition": condition,
        "summary": summary,
        "out_dir": str(out_dir),
    }

# %%
# ============================================================
# 8. Run one condition first, then all configured conditions
# ============================================================
# Execution is intentionally deferred to main() below.

def aggregate_metrics():
    frames = []
    missing = []

    for cond in RUN_CONDITIONS:
        p = OUTPUT_ROOT / f"{cond}_NPC_to_Neuron" / "metrics_summary.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
        else:
            missing.append(cond)

    if not frames:
        raise FileNotFoundError(
            "No condition-level metrics_summary.csv files were found."
        )

    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(
        OUTPUT_ROOT / "all_conditions_metrics_summary.csv",
        index=False,
    )

    print(summary.to_string(index=False))
    if missing:
        print(f"\nNot yet run: {missing}")

    return summary


def main():
    if _ARGS.aggregate_only:
        return aggregate_metrics()
    conditions = RUN_CONDITIONS if _ARGS.all else [_ARGS.condition or DEBUG_CONDITION]
    for condition in conditions:
        run_condition(condition)
    return aggregate_metrics()

if __name__ == '__main__':
    main()

