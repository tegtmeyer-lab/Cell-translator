"""Training utilities for validation-specific LOLO morphology models."""
from __future__ import annotations
import copy
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .models import LodoMorphologyAutoencoder, build_lodo_generator, build_lodo_discriminator


class LodoNumpyDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(np.asarray(X, np.float32), dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx]


def set_all_seeds(seed=42, include_tensorflow=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if include_tensorflow:
        import tensorflow as tf
        tf.random.set_seed(seed)


def lodo_morphology_loss(recon, x):
    recon_loss = torch.mean((recon - x) ** 2)
    mean_loss = torch.mean((recon.mean(dim=0) - x.mean(dim=0)) ** 2)
    std_loss = torch.mean((recon.std(dim=0) - x.std(dim=0)) ** 2)
    return recon_loss + 0.05 * mean_loss + 0.40 * std_loss


def fit_lodo_ae(df, feature_cols, out_prefix, cfg, seed=42):
    """Fit median imputer, StandardScaler and AE using training lines only."""
    set_all_seeds(seed, include_tensorflow=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    idx = np.arange(len(X_raw))
    tr, va = train_test_split(
        idx, test_size=float(cfg["validation_fraction"]), random_state=seed, shuffle=True
    )
    median = X_raw.iloc[tr].median(axis=0)
    X_imp = X_raw.fillna(median)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_imp.iloc[tr]).astype(np.float32)
    X_va = scaler.transform(X_imp.iloc[va]).astype(np.float32)
    X_all = scaler.transform(X_imp).astype(np.float32)

    model = LodoMorphologyAutoencoder(
        input_dim=X_all.shape[1], latent_dim=int(cfg["latent_dim"]),
        dropout=float(cfg.get("dropout", 0.03))
    ).to(device)
    train_dl = DataLoader(
        LodoNumpyDataset(X_tr), batch_size=int(cfg["batch_size"]),
        shuffle=True, drop_last=False
    )
    val_tensor = torch.tensor(X_va, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=1e-5)

    best_val, best_state, best_epoch, wait = float("inf"), None, -1, 0
    history = {"train_loss": [], "val_mse": []}
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train(); total = 0.0
        for batch in train_dl:
            batch = batch.to(device)
            _, recon = model(batch)
            loss = lodo_morphology_loss(recon, batch)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step(); total += float(loss.item())
        train_loss = total / max(len(train_dl), 1)
        model.eval()
        with torch.no_grad():
            _, recon_va = model(val_tensor)
            val_mse = float(F.mse_loss(recon_va, val_tensor).item())
        history["train_loss"].append(train_loss); history["val_mse"].append(val_mse)
        if val_mse < best_val:
            best_val = val_mse; best_epoch = epoch; wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= int(cfg["patience"]): break
    if best_state is not None: model.load_state_dict(best_state)

    Z_all = encode_lodo(model, X_all, device)
    out_prefix = Path(out_prefix); out_prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(), "input_dim": X_all.shape[1],
        "latent_dim": int(cfg["latent_dim"]), "feature_cols": list(feature_cols),
        "train_idx": tr, "val_idx": va, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
        "train_median": median.to_numpy(), "best_epoch": best_epoch, "best_val": best_val,
        "history": history,
    }, str(out_prefix) + "_ae_checkpoint.pt")
    np.save(str(out_prefix) + "_X_all_scaled.npy", X_all)
    np.save(str(out_prefix) + "_Z_all.npy", Z_all)
    return {"model": model, "device": device, "scaler": scaler, "train_median": median,
            "X_all_scaled": X_all, "Z_all": Z_all, "best_epoch": best_epoch, "best_val": best_val}


def transform_with_lodo_scaler(df, feature_cols, train_median, scaler):
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(train_median)
    X_orig = X.to_numpy(np.float32)
    return scaler.transform(X_orig).astype(np.float32), X_orig


def encode_lodo(model, X, device, batch_size=512):
    loader = DataLoader(LodoNumpyDataset(X), batch_size=batch_size, shuffle=False)
    z = []; model.eval()
    with torch.no_grad():
        for batch in loader:
            z.append(model.encode(batch.to(device)).cpu().numpy())
    return np.concatenate(z, axis=0).astype(np.float32)


def decode_lodo(model, Z, device, batch_size=512):
    loader = DataLoader(LodoNumpyDataset(Z), batch_size=batch_size, shuffle=False)
    out = []; model.eval()
    with torch.no_grad():
        for batch in loader:
            out.append(model.decode(batch.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def standardize_latent(Z_fit, Z_transform, eps=1e-6):
    mean = Z_fit.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(Z_fit.std(axis=0, keepdims=True), eps).astype(np.float32)
    return ((Z_transform - mean) / std).astype(np.float32), mean, std


def _gradient_penalty(discriminator, real, fake):
    import tensorflow as tf
    alpha = tf.random.uniform([tf.shape(real)[0], 1])
    inter = real + alpha * (fake - real)
    with tf.GradientTape() as tape:
        tape.watch(inter); pred = discriminator(inter, training=True)
    grad = tape.gradient(pred, inter)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1) + 1e-12)
    return tf.reduce_mean((norm - 1.0) ** 2)


def _moment_loss(fake, real):
    import tensorflow as tf
    return tf.reduce_mean(tf.abs(tf.reduce_mean(fake, 0) - tf.reduce_mean(real, 0))) + \
        tf.reduce_mean(tf.abs(tf.math.reduce_std(fake, 0) - tf.math.reduce_std(real, 0)))


def train_lodo_cyclegan(train_src_z, train_tgt_z, fold_dir, source_name, target_name, cfg, seed=42):
    """Train the reduced LOLO latent CycleGAN used in the validation notebook."""
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    set_all_seeds(seed, include_tensorflow=True)

    src_std, mean_src, std_src = standardize_latent(train_src_z, train_src_z)
    tgt_std, mean_tgt, std_tgt = standardize_latent(train_tgt_z, train_tgt_z)
    d_src, d_tgt = src_std.shape[1], tgt_std.shape[1]
    if d_src != d_tgt: raise ValueError("Source/target latent dimensions must match")

    Gf = build_lodo_generator(d_src, d_tgt, f"G_{source_name}_to_{target_name}")
    Gb = build_lodo_generator(d_tgt, d_src, f"G_{target_name}_to_{source_name}")
    Dt = build_lodo_discriminator(d_tgt, f"D_{target_name}")
    Ds = build_lodo_discriminator(d_src, f"D_{source_name}")
    opts = [Adam(1e-4, beta_1=0.5, clipnorm=1.0) for _ in range(4)]
    opt_gf, opt_gb, opt_dt, opt_ds = opts

    batch = int(cfg["batch_size"]); n_critic = int(cfg["n_critic"])
    steps = max(min(len(src_std), len(tgt_std)) // batch, 1)
    ds_src = tf.data.Dataset.from_tensor_slices(src_std).shuffle(4096, seed=seed).repeat()
    ds_tgt = tf.data.Dataset.from_tensor_slices(tgt_std).shuffle(4096, seed=seed + 1).repeat()
    ds = tf.data.Dataset.zip((ds_src, ds_tgt)).batch(batch, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    gp_w, cyc_w, id_w, mom_w = 10.0, 5.0, 0.5, 2.0

    @tf.function
    def step(real_src, real_tgt):
        for _ in tf.range(n_critic):
            with tf.GradientTape(persistent=True) as td:
                fake_tgt = Gf(real_src, training=True); fake_src = Gb(real_tgt, training=True)
                rt = Dt(real_tgt, training=True); ft = Dt(fake_tgt, training=True)
                rs = Ds(real_src, training=True); fs = Ds(fake_src, training=True)
                lt = tf.reduce_mean(ft) - tf.reduce_mean(rt) + gp_w * _gradient_penalty(Dt, real_tgt, fake_tgt)
                ls = tf.reduce_mean(fs) - tf.reduce_mean(rs) + gp_w * _gradient_penalty(Ds, real_src, fake_src)
            opt_dt.apply_gradients(zip(td.gradient(lt, Dt.trainable_variables), Dt.trainable_variables))
            opt_ds.apply_gradients(zip(td.gradient(ls, Ds.trainable_variables), Ds.trainable_variables))
            del td
        with tf.GradientTape(persistent=True) as tg:
            fake_tgt = Gf(real_src, training=True); fake_src = Gb(real_tgt, training=True)
            cyc_src = Gb(fake_tgt, training=True); cyc_tgt = Gf(fake_src, training=True)
            id_tgt = Gf(real_tgt, training=True); id_src = Gb(real_src, training=True)
            cycle = tf.reduce_mean(tf.abs(real_src-cyc_src)) + tf.reduce_mean(tf.abs(real_tgt-cyc_tgt))
            identity = tf.reduce_mean(tf.abs(real_tgt-id_tgt)) + tf.reduce_mean(tf.abs(real_src-id_src))
            mom_f = _moment_loss(fake_tgt, real_tgt); mom_b = _moment_loss(fake_src, real_src)
            lgf = -tf.reduce_mean(Dt(fake_tgt, training=True)) + cyc_w*cycle + id_w*tf.reduce_mean(tf.abs(real_tgt-id_tgt)) + mom_w*mom_f
            lgb = -tf.reduce_mean(Ds(fake_src, training=True)) + cyc_w*cycle + id_w*tf.reduce_mean(tf.abs(real_src-id_src)) + mom_w*mom_b
        opt_gf.apply_gradients(zip(tg.gradient(lgf, Gf.trainable_variables), Gf.trainable_variables))
        opt_gb.apply_gradients(zip(tg.gradient(lgb, Gb.trainable_variables), Gb.trainable_variables)); del tg
        return lgf, lgb, lt, ls, cycle, identity, mom_f

    history = {k: [] for k in ["G_forward","G_backward","D_target","D_source","cycle","identity","moment"]}
    for _epoch in range(1, int(cfg["epochs"]) + 1):
        acc = np.zeros(7, dtype=float); n=0
        for a,b in ds.take(steps):
            vals = step(a,b); acc += np.array([float(v) for v in vals]); n += 1
        vals = acc / max(n,1)
        for k,v in zip(history, vals): history[k].append(float(v))

    fold_dir = Path(fold_dir); fold_dir.mkdir(parents=True, exist_ok=True)
    Gf.save(fold_dir / "G_forward.keras"); Gb.save(fold_dir / "G_backward.keras")
    np.savez_compressed(fold_dir / "cyclegan_history.npz", **{k: np.asarray(v) for k,v in history.items()})
    np.savez_compressed(fold_dir / "latent_standardization.npz",
                        mean_src=mean_src, std_src=std_src, mean_tgt=mean_tgt, std_tgt=std_tgt)
    return {"G_forward": Gf, "G_backward": Gb, "mean_src": mean_src, "std_src": std_src,
            "mean_tgt": mean_tgt, "std_tgt": std_tgt, "history": history}
