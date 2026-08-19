"""Historical latent CycleGAN training variant used in acute leukemia analyses.

The generator and critic architectures are imported from the shared Cell
Translator model package. This module preserves the acute-leukemia-specific
objective weights and paired/truncated training schedule used for the reported
CD34+ -> BMMC and CD34+ -> MPAL analyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.optimizers import Adam

from cell_translator.models.cyclegan import (
    build_discriminator,
    build_generator,
    build_generator_with_affine_head,
)


@dataclass(frozen=True)
class AcuteLeukemiaCycleGANConfig:
    batch_size: int = 128
    epochs: int = 300
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.1
    gp_weight: float = 10.0
    lambda_sup: float = 0.0
    k_pair: int = 9
    bank_sample: int = 5000
    n_critic: int = 3
    drift_eps: float = 1e-3
    pair_weight: float = 5.0
    mmd_weight: float = 2.0
    covariance_weight: float = 2.0
    nce_weight: float = 0.0
    nce_tau: float = 0.07
    mmd_sigmas: tuple[float, ...] = (1.0, 2.0, 4.0)
    learning_rate: float = 1e-4
    lr_decay_steps: int = 10000
    lr_decay_rate: float = 0.9
    adam_beta1: float = 0.5
    clipnorm: float = 1.0
    shuffle_buffer: int = 1024
    mnn_every: int = 10
    mnn_pool: int = 4000
    tf_seed: int | None = None


HISTORY_KEYS = (
    "gen_fwd_loss", "gen_bwd_loss", "disc_tgt_loss", "disc_src_loss",
    "cycle_loss", "identity_loss", "pair_loss", "mmd_loss", "sup_loss",
    "wasserstein_gap",
)


def generator_loss(fake_scores):
    return -tf.reduce_mean(fake_scores)


def discriminator_loss(real_scores, fake_scores):
    return tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)


def cycle_loss(a, b):
    return tf.reduce_mean(tf.abs(a - b))


def identity_loss(a, b):
    return tf.reduce_mean(tf.abs(a - b))


def gradient_penalty(discriminator, real, fake):
    alpha = tf.random.uniform([tf.shape(real)[0], 1])
    inter = real + alpha * (fake - real)
    with tf.GradientTape() as tape:
        tape.watch(inter)
        pred = discriminator(inter)
    grad = tape.gradient(pred, inter)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1) + 1e-12)
    return tf.reduce_mean((norm - 1.0) ** 2)


def knn_pair_loss_cos(fake_tgt, tgt_bank, k=9, bank_sample=5000):
    n_bank = tf.shape(tgt_bank)[0]
    m = tf.minimum(bank_sample, n_bank)
    idx = tf.random.uniform([m], 0, n_bank, dtype=tf.int32)
    bank = tf.gather(tgt_bank, idx)
    f = tf.math.l2_normalize(fake_tgt, axis=1)
    b = tf.math.l2_normalize(bank, axis=1)
    sims = tf.matmul(f, b, transpose_b=True)
    _, inds = tf.math.top_k(sims, k=k)
    b_knn = tf.gather(b, inds, axis=0)
    b_tgt = tf.reduce_mean(b_knn, axis=1)
    return tf.reduce_mean(1.0 - tf.reduce_sum(f * b_tgt, axis=1))


def covariance_loss(fake, real):
    def cov(x):
        xc = x - tf.reduce_mean(x, axis=0, keepdims=True)
        n = tf.cast(tf.shape(xc)[0] - 1, tf.float32)
        return tf.matmul(xc, xc, transpose_a=True) / n
    return tf.reduce_mean(tf.square(cov(fake) - cov(real)))


def info_nce(fake, real, tau=0.07):
    f = tf.math.l2_normalize(fake, axis=1)
    r = tf.math.l2_normalize(real, axis=1)
    logits = tf.matmul(f, r, transpose_b=True) / tau
    pos_idx = tf.argmax(tf.stop_gradient(logits), axis=1)
    pos_onehot = tf.one_hot(pos_idx, depth=tf.shape(logits)[1])
    logsumexp = tf.reduce_logsumexp(logits, axis=1)
    pos_logit = tf.reduce_sum(logits * pos_onehot, axis=1)
    return tf.reduce_mean(-(pos_logit - logsumexp))


def mmd_rbf(x, y, sigmas=(1.0, 2.0, 4.0)):
    def kernel(a, b, s):
        gamma = 1.0 / (2.0 * s * s)
        a2 = tf.reduce_sum(a * a, 1, keepdims=True)
        b2 = tf.reduce_sum(b * b, 1, keepdims=True)
        d2 = a2 - 2.0 * tf.matmul(a, b, transpose_b=True) + tf.transpose(b2)
        return tf.exp(-gamma * d2)
    kxx = tf.add_n([kernel(x, x, s) for s in sigmas]) / len(sigmas)
    kyy = tf.add_n([kernel(y, y, s) for s in sigmas]) / len(sigmas)
    kxy = tf.add_n([kernel(x, y, s) for s in sigmas]) / len(sigmas)
    return tf.reduce_mean(kxx) + tf.reduce_mean(kyy) - 2.0 * tf.reduce_mean(kxy)


def make_mnn_pairs(z_fake_np, z_real_np):
    mask_f = np.isfinite(z_fake_np).all(axis=1)
    mask_r = np.isfinite(z_real_np).all(axis=1)
    z_fake_np = z_fake_np[mask_f]
    z_real_np = z_real_np[mask_r]
    if len(z_fake_np) == 0 or len(z_real_np) == 0:
        return np.array([], int), np.array([], int)
    nbr_r = NearestNeighbors(n_neighbors=1).fit(z_real_np)
    idx_rf = nbr_r.kneighbors(z_fake_np, return_distance=False)[:, 0]
    nbr_f = NearestNeighbors(n_neighbors=1).fit(z_fake_np)
    idx_fr = nbr_f.kneighbors(z_real_np, return_distance=False)[:, 0]
    pairs = [(i, j) for i, j in enumerate(idx_rf) if idx_fr[j] == i]
    if not pairs:
        return np.array([], int), np.array([], int)
    return np.array([i for i, _ in pairs]), np.array([j for _, j in pairs])


def make_acute_leukemia_train_step(
    g_fwd, g_bwd, d_tgt, d_src,
    opt_g_fwd, opt_g_bwd, opt_d_tgt, opt_d_src,
    tgt_bank, config: AcuteLeukemiaCycleGANConfig,
):
    @tf.function
    def train_step(real_src, real_tgt, pair_src, pair_tgt):
        for _ in tf.range(config.n_critic):
            with tf.GradientTape(persistent=True) as tape_d:
                fake_tgt = g_fwd(real_src, training=True)
                fake_src = g_bwd(real_tgt, training=True)
                r_tgt = d_tgt(real_tgt, training=True)
                f_tgt = d_tgt(fake_tgt, training=True)
                r_src = d_src(real_src, training=True)
                f_src = d_src(fake_src, training=True)
                gp_tgt = gradient_penalty(d_tgt, real_tgt, fake_tgt)
                gp_src = gradient_penalty(d_src, real_src, fake_src)
                drift_tgt = config.drift_eps * (
                    tf.reduce_mean(tf.square(r_tgt)) + tf.reduce_mean(tf.square(f_tgt))
                )
                drift_src = config.drift_eps * (
                    tf.reduce_mean(tf.square(r_src)) + tf.reduce_mean(tf.square(f_src))
                )
                disc_tgt = discriminator_loss(r_tgt, f_tgt) + config.gp_weight * gp_tgt + drift_tgt
                disc_src = discriminator_loss(r_src, f_src) + config.gp_weight * gp_src + drift_src
            opt_d_tgt.apply_gradients(zip(tape_d.gradient(disc_tgt, d_tgt.trainable_variables), d_tgt.trainable_variables))
            opt_d_src.apply_gradients(zip(tape_d.gradient(disc_src, d_src.trainable_variables), d_src.trainable_variables))
            del tape_d

        w_gap = tf.reduce_mean(r_tgt) - tf.reduce_mean(f_tgt)
        with tf.GradientTape(persistent=True) as tape_g:
            fake_tgt = g_fwd(real_src, training=True)
            fake_src = g_bwd(real_tgt, training=True)
            cycled_src = g_bwd(fake_tgt, training=True)
            cycled_tgt = g_fwd(fake_src, training=True)
            id_tgt = g_fwd(real_tgt, training=True)
            id_src = g_bwd(real_src, training=True)

            gen_fwd = generator_loss(d_tgt(fake_tgt, training=True))
            gen_bwd = generator_loss(d_src(fake_src, training=True))
            cyc_src = cycle_loss(real_src, cycled_src)
            cyc_tgt = cycle_loss(real_tgt, cycled_tgt)
            id_fwd = identity_loss(real_tgt, id_tgt)
            id_bwd = identity_loss(real_src, id_src)
            pair = knn_pair_loss_cos(fake_tgt, tgt_bank, config.k_pair, config.bank_sample)
            mmd = mmd_rbf(fake_tgt, real_tgt, config.mmd_sigmas)
            cov = covariance_loss(fake_tgt, real_tgt)
            nce = info_nce(fake_tgt, real_tgt, config.nce_tau)

            def sup_term():
                pred = g_fwd(pair_src, training=True)
                return tf.reduce_mean(tf.square(pred - pair_tgt))
            sup = tf.cond(
                tf.shape(pair_src)[0] > 0,
                sup_term,
                lambda: tf.constant(0.0, tf.float32),
            )
            total_fwd = (
                gen_fwd
                + config.lambda_cycle * (cyc_src + cyc_tgt)
                + config.lambda_identity * id_fwd
                + config.pair_weight * pair
                + config.mmd_weight * mmd
                + config.covariance_weight * cov
                + config.nce_weight * nce
                + config.lambda_sup * sup
            )
            total_bwd = (
                gen_bwd
                + config.lambda_cycle * (cyc_src + cyc_tgt)
                + config.lambda_identity * id_bwd
            )

        opt_g_fwd.apply_gradients(zip(tape_g.gradient(total_fwd, g_fwd.trainable_variables), g_fwd.trainable_variables))
        opt_g_bwd.apply_gradients(zip(tape_g.gradient(total_bwd, g_bwd.trainable_variables), g_bwd.trainable_variables))
        del tape_g
        return {
            "gen_fwd_loss": total_fwd,
            "gen_bwd_loss": total_bwd,
            "disc_tgt_loss": disc_tgt,
            "disc_src_loss": disc_src,
            "cycle_loss": cyc_src + cyc_tgt,
            "identity_loss": id_fwd + id_bwd,
            "pair_loss": pair,
            "mmd_loss": mmd,
            "sup_loss": sup,
            "wasserstein_gap": w_gap,
        }
    return train_step


def train_acute_leukemia_cyclegan(
    train_src: np.ndarray,
    train_tgt: np.ndarray,
    source_name: str,
    target_name: str,
    config: AcuteLeukemiaCycleGANConfig,
) -> dict[str, Any]:
    """Train the acute-leukemia CycleGAN variant on standardized latents."""
    tf.keras.backend.clear_session()
    if config.tf_seed is not None:
        tf.keras.utils.set_random_seed(config.tf_seed)

    train_src = np.asarray(train_src, dtype=np.float32)
    train_tgt = np.asarray(train_tgt, dtype=np.float32)
    if train_src.ndim != 2 or train_tgt.ndim != 2 or train_src.shape[1] != train_tgt.shape[1]:
        raise ValueError(f"Latent shape mismatch: {train_src.shape} vs {train_tgt.shape}")

    # Historical implementation balances by truncating both arrays to the
    # smaller domain before constructing a paired shuffled dataset.
    n = min(train_src.shape[0], train_tgt.shape[0])
    src = train_src[:n]
    tgt = train_tgt[:n]
    ds = tf.data.Dataset.from_tensor_slices((src, tgt))
    if config.tf_seed is None:
        ds = ds.shuffle(config.shuffle_buffer)
    else:
        ds = ds.shuffle(config.shuffle_buffer, seed=config.tf_seed, reshuffle_each_iteration=True)
    ds = ds.batch(config.batch_size, drop_remainder=True)

    dim = src.shape[1]
    g_fwd = build_generator_with_affine_head(dim, dim, f"G_{source_name}_to_{target_name}")
    g_bwd = build_generator(dim, dim, f"G_{target_name}_to_{source_name}")
    d_tgt = build_discriminator(dim, f"D_{target_name}")
    d_src = build_discriminator(dim, f"D_{source_name}")
    tgt_bank = tf.constant(tgt, dtype=tf.float32)

    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        config.learning_rate,
        decay_steps=config.lr_decay_steps,
        decay_rate=config.lr_decay_rate,
        staircase=True,
    )
    opts = [Adam(learning_rate=lr, beta_1=config.adam_beta1, clipnorm=config.clipnorm) for _ in range(4)]
    step_fn = make_acute_leukemia_train_step(
        g_fwd, g_bwd, d_tgt, d_src, *opts, tgt_bank, config
    )

    histories = {k: [] for k in HISTORY_KEYS}
    pair_src = tf.zeros([0, dim], tf.float32)
    pair_tgt = tf.zeros([0, dim], tf.float32)

    for epoch in range(1, config.epochs + 1):
        if epoch % config.mnn_every == 1 and config.lambda_sup > 0.0:
            pool = min(config.mnn_pool, src.shape[0], tgt.shape[0])
            idx_s = np.random.choice(src.shape[0], pool, replace=False)
            idx_t = np.random.choice(tgt.shape[0], pool, replace=False)
            z_s, z_t = src[idx_s], tgt[idx_t]
            z_f = g_fwd(tf.constant(z_s), training=False).numpy()
            z_f = np.nan_to_num(z_f, nan=0.0, posinf=1e6, neginf=-1e6)
            z_t = np.nan_to_num(z_t, nan=0.0, posinf=1e6, neginf=-1e6)
            I, J = make_mnn_pairs(z_f, z_t)
            if len(I):
                pair_src = tf.constant(z_s[I], tf.float32)
                pair_tgt = tf.constant(z_t[J], tf.float32)

        totals = {k: 0.0 for k in HISTORY_KEYS}
        steps = 0
        for real_src, real_tgt in ds:
            out = step_fn(real_src, real_tgt, pair_src, pair_tgt)
            for key in totals:
                totals[key] += float(out[key])
            steps += 1
        if steps == 0:
            raise ValueError("No CycleGAN batches were produced; reduce batch size or provide more cells.")
        for key in histories:
            histories[key].append(totals[key] / steps)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{source_name}->{target_name}] Epoch {epoch:03d} | "
                f"W-gap {histories['wasserstein_gap'][-1]:.3f} | "
                f"gen {histories['gen_fwd_loss'][-1]:.3f} | "
                f"disc {histories['disc_tgt_loss'][-1]:.3f} | "
                f"pair {histories['pair_loss'][-1]:.3f} | "
                f"mmd {histories['mmd_loss'][-1]:.3f} | "
                f"cycle {histories['cycle_loss'][-1]:.3f} | "
                f"id {histories['identity_loss'][-1]:.3f}"
            )

    return {
        "G_fwd": g_fwd, "G_bwd": g_bwd, "D_tgt": d_tgt, "D_src": d_src,
        "histories": histories, "n_balanced_training_cells": int(n),
    }
