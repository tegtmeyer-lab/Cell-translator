"""Latent-space WGAN-GP CycleGAN losses and training routine.

This module contains the shared CycleGAN implementation used across Cell
Translator applications. Application-specific training settings are supplied
through ``CycleGANTrainingConfig`` so each reported analysis can retain its
original parameterization without redefining the model or loss functions.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.optimizers import Adam

from cell_translator.hyperparameters import (
    ADAM_BETA1,
    BANK_SAMPLE,
    CGAN_BATCH_SIZE,
    CGAN_EPOCHS,
    CLIPNORM,
    COV_W,
    DRIFT_EPS,
    GP_WEIGHT,
    HISTORY_KEYS,
    K_PAIR,
    LAMBDA_CYCLE,
    LAMBDA_IDENTITY,
    LAMBDA_SUP,
    LR_DECAY_RATE,
    LR_DECAY_STEPS,
    LR_INITIAL,
    MMD_W,
    MNN_EVERY,
    MNN_POOL,
    MOM_W,
    N_CRITIC,
    PAIR_W,
    SEED,
)
from cell_translator.models.cyclegan import (
    build_discriminator,
    build_generator,
    build_generator_with_affine_head,
)
from cell_translator.utils.reproducibility import set_tf_seed


@dataclass(frozen=True)
class CycleGANTrainingConfig:
    """CycleGAN settings with neuronal-analysis defaults.

    Defaults reproduce the canonical neuronal application. The perturbation
    YAML files explicitly override only the settings that differed in the
    original perturbation notebook.
    """

    batch_size: int = CGAN_BATCH_SIZE
    epochs: int = CGAN_EPOCHS
    gp_weight: float = GP_WEIGHT
    drift_eps: float = DRIFT_EPS
    k_pair: int = K_PAIR
    bank_sample: int = BANK_SAMPLE
    lambda_sup: float = LAMBDA_SUP
    lambda_cycle: float = LAMBDA_CYCLE
    lambda_identity: float = LAMBDA_IDENTITY
    pair_w: float = PAIR_W
    mmd_w: float = MMD_W
    cov_w: float = COV_W
    mom_w: float = MOM_W
    n_critic: int = N_CRITIC
    lr_initial: float = LR_INITIAL
    lr_decay_steps: int = LR_DECAY_STEPS
    lr_decay_rate: float = LR_DECAY_RATE
    adam_beta1: float = ADAM_BETA1
    clipnorm: float = CLIPNORM
    mnn_every: int = MNN_EVERY
    mnn_pool: int = MNN_POOL
    mmd_sigmas: tuple[float, ...] = (1.0, 2.0, 4.0)
    moment_eps_mode: str = "symmetric"
    source_shuffle_buffer: int = 4096
    target_shuffle_buffer: int = 4096
    steps_per_epoch_from: str = "target"  # "source" or "target"
    mnn_target_replace: bool = False
    tf_seed: Optional[int] = SEED

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]] = None):
        if not values:
            return cls()
        fields = cls.__dataclass_fields__
        unknown = set(values) - set(fields)
        if unknown:
            raise KeyError(f"Unknown CycleGAN settings: {sorted(unknown)}")
        values = dict(values)
        if "mmd_sigmas" in values:
            values["mmd_sigmas"] = tuple(float(x) for x in values["mmd_sigmas"])
        return cls(**values)


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


def knn_pair_loss_cos(fake_tgt, tgt_bank, k=5, bank_sample=8192):
    """Cosine k-NN pull toward the real target latent manifold."""
    n_bank = tf.shape(tgt_bank)[0]
    m = tf.minimum(bank_sample, n_bank)
    idx = tf.random.uniform([m], 0, n_bank, dtype=tf.int32)
    bank = tf.gather(tgt_bank, idx)
    fake_norm = tf.math.l2_normalize(fake_tgt, axis=1)
    bank_norm = tf.math.l2_normalize(bank, axis=1)
    sims = tf.matmul(fake_norm, bank_norm, transpose_b=True)
    _, inds = tf.math.top_k(sims, k=k)
    bank_knn = tf.gather(bank_norm, inds, axis=0)
    bank_target = tf.reduce_mean(bank_knn, axis=1)
    return tf.reduce_mean(1.0 - tf.reduce_sum(fake_norm * bank_target, axis=1))


def cov_loss(fake, real):
    """Frobenius covariance alignment."""

    def cov(x):
        xc = x - tf.reduce_mean(x, axis=0, keepdims=True)
        n = tf.cast(tf.shape(xc)[0] - 1, tf.float32)
        return tf.matmul(xc, xc, transpose_a=True) / n

    return tf.reduce_mean(tf.square(cov(fake) - cov(real)))


def moment_loss(fake, real, eps=1e-6, eps_mode: str = "symmetric"):
    """Mean + standard-deviation matching per latent dimension.

    ``symmetric`` adds epsilon to both standard deviations (the canonical
    shared implementation). ``difference`` preserves the historical organoid
    notebook expression ``abs(std_fake - std_real + eps)``.
    """
    mf = tf.reduce_mean(fake, axis=0)
    mr = tf.reduce_mean(real, axis=0)
    sf = tf.math.reduce_std(fake, axis=0)
    sr = tf.math.reduce_std(real, axis=0)
    if eps_mode == "symmetric":
        std_term = tf.abs((sf + eps) - (sr + eps))
    elif eps_mode == "difference":
        std_term = tf.abs(sf - sr + eps)
    else:
        raise ValueError("eps_mode must be 'symmetric' or 'difference'.")
    return tf.reduce_mean(tf.abs(mf - mr)) + tf.reduce_mean(std_term)


def mmd_rbf(x, y, sigmas: Sequence[float] = (1.0, 2.0, 4.0)):
    """Multi-scale RBF maximum mean discrepancy."""

    def kernel(a, b, sigma):
        gamma = 1.0 / (2 * sigma * sigma)
        a2 = tf.reduce_sum(a * a, 1, keepdims=True)
        b2 = tf.reduce_sum(b * b, 1, keepdims=True)
        d2 = a2 - 2.0 * tf.matmul(a, b, transpose_b=True) + tf.transpose(b2)
        return tf.exp(-gamma * d2)

    kxx = tf.add_n([kernel(x, x, s) for s in sigmas]) / len(sigmas)
    kyy = tf.add_n([kernel(y, y, s) for s in sigmas]) / len(sigmas)
    kxy = tf.add_n([kernel(x, y, s) for s in sigmas]) / len(sigmas)
    return tf.reduce_mean(kxx) + tf.reduce_mean(kyy) - 2.0 * tf.reduce_mean(kxy)


def make_mnn_pairs(Z_fake_np, Z_real_np):
    """Mutual nearest neighbors between fake and real target latents."""
    mask_f = np.isfinite(Z_fake_np).all(axis=1)
    mask_r = np.isfinite(Z_real_np).all(axis=1)
    Z_fake_np = Z_fake_np[mask_f]
    Z_real_np = Z_real_np[mask_r]
    if len(Z_fake_np) == 0 or len(Z_real_np) == 0:
        return np.array([], int), np.array([], int)

    nbr_r = NearestNeighbors(n_neighbors=1).fit(Z_real_np)
    idx_rf = nbr_r.kneighbors(Z_fake_np, return_distance=False)[:, 0]
    nbr_f = NearestNeighbors(n_neighbors=1).fit(Z_fake_np)
    idx_fr = nbr_f.kneighbors(Z_real_np, return_distance=False)[:, 0]
    mnn = [(i, j) for i, j in enumerate(idx_rf) if idx_fr[j] == i]
    if not mnn:
        return np.array([], int), np.array([], int)
    return (
        np.array([i for i, _ in mnn], dtype=int),
        np.array([j for _, j in mnn], dtype=int),
    )


def make_cyclegan_train_step(
    G_fwd,
    G_bwd,
    D_tgt,
    D_src,
    opt_G_fwd,
    opt_G_bwd,
    opt_D_tgt,
    opt_D_src,
    tgt_bank,
    config: CycleGANTrainingConfig | None = None,
):
    """Build the shared ``@tf.function`` CycleGAN training step."""
    cfg = config or CycleGANTrainingConfig()

    @tf.function
    def train_step(real_src, real_tgt, pair_src, pair_tgt, n_critic=tf.constant(N_CRITIC)):
        for _ in tf.range(n_critic):
            with tf.GradientTape(persistent=True) as tD:
                fake_tgt = G_fwd(real_src, training=True)
                fake_src = G_bwd(real_tgt, training=True)

                r_tgt = D_tgt(real_tgt, training=True)
                f_tgt = D_tgt(fake_tgt, training=True)
                r_src = D_src(real_src, training=True)
                f_src = D_src(fake_src, training=True)

                gp_tgt = gradient_penalty(D_tgt, real_tgt, fake_tgt)
                gp_src = gradient_penalty(D_src, real_src, fake_src)
                drift_tgt = cfg.drift_eps * (
                    tf.reduce_mean(tf.square(r_tgt)) + tf.reduce_mean(tf.square(f_tgt))
                )
                drift_src = cfg.drift_eps * (
                    tf.reduce_mean(tf.square(r_src)) + tf.reduce_mean(tf.square(f_src))
                )
                disc_tgt_loss = (
                    discriminator_loss(r_tgt, f_tgt) + cfg.gp_weight * gp_tgt + drift_tgt
                )
                disc_src_loss = (
                    discriminator_loss(r_src, f_src) + cfg.gp_weight * gp_src + drift_src
                )

            d_tgt_grads = tD.gradient(disc_tgt_loss, D_tgt.trainable_variables)
            d_src_grads = tD.gradient(disc_src_loss, D_src.trainable_variables)
            opt_D_tgt.apply_gradients(zip(d_tgt_grads, D_tgt.trainable_variables))
            opt_D_src.apply_gradients(zip(d_src_grads, D_src.trainable_variables))
            del tD

        w_gap = tf.reduce_mean(r_tgt) - tf.reduce_mean(f_tgt)

        with tf.GradientTape(persistent=True) as tape_g:
            fake_tgt = G_fwd(real_src, training=True)
            fake_src = G_bwd(real_tgt, training=True)
            gen_fwd_adv = generator_loss(D_tgt(fake_tgt, training=True))
            gen_bwd_adv = generator_loss(D_src(fake_src, training=True))

            cycled_src = G_bwd(fake_tgt, training=True)
            cycled_tgt = G_fwd(fake_src, training=True)
            cyc_src = cycle_loss(real_src, cycled_src)
            cyc_tgt = cycle_loss(real_tgt, cycled_tgt)

            id_tgt = G_fwd(real_tgt, training=True)
            id_src = G_bwd(real_src, training=True)
            id_loss_fwd = identity_loss(real_tgt, id_tgt)
            id_loss_bwd = identity_loss(real_src, id_src)

            pair = knn_pair_loss_cos(
                fake_tgt, tgt_bank, k=cfg.k_pair, bank_sample=cfg.bank_sample
            )
            mmd = mmd_rbf(fake_tgt, real_tgt, sigmas=cfg.mmd_sigmas)
            cval = cov_loss(fake_tgt, real_tgt)
            mom = moment_loss(fake_tgt, real_tgt, eps_mode=cfg.moment_eps_mode)
            mom_bwd = moment_loss(fake_src, real_src, eps_mode=cfg.moment_eps_mode)

            def sup_term():
                pred = G_fwd(pair_src, training=True)
                return tf.reduce_mean(tf.square(pred - pair_tgt))

            sup = tf.cond(
                tf.logical_and(cfg.lambda_sup > 0.0, tf.shape(pair_src)[0] > 0),
                sup_term,
                lambda: tf.constant(0.0, tf.float32),
            )

            total_gen_fwd_loss = (
                gen_fwd_adv
                + cfg.lambda_cycle * (cyc_src + cyc_tgt)
                + cfg.lambda_identity * id_loss_fwd
                + cfg.pair_w * pair
                + cfg.mmd_w * mmd
                + cfg.cov_w * cval
                + cfg.mom_w * mom
                + cfg.lambda_sup * sup
            )
            total_gen_bwd_loss = (
                gen_bwd_adv
                + cfg.lambda_cycle * (cyc_src + cyc_tgt)
                + cfg.lambda_identity * id_loss_bwd
                + cfg.mom_w * mom_bwd
            )

        g_fwd_grads = tape_g.gradient(total_gen_fwd_loss, G_fwd.trainable_variables)
        g_bwd_grads = tape_g.gradient(total_gen_bwd_loss, G_bwd.trainable_variables)
        opt_G_fwd.apply_gradients(zip(g_fwd_grads, G_fwd.trainable_variables))
        opt_G_bwd.apply_gradients(zip(g_bwd_grads, G_bwd.trainable_variables))
        del tape_g

        return {
            "gen_fwd_loss": total_gen_fwd_loss,
            "gen_bwd_loss": total_gen_bwd_loss,
            "disc_tgt_loss": disc_tgt_loss,
            "disc_src_loss": disc_src_loss,
            "cycle_loss": cyc_src + cyc_tgt,
            "identity_loss": id_loss_fwd + id_loss_bwd,
            "pair_loss": pair,
            "mmd_loss": mmd,
            "cov_loss": cval,
            "mom_loss": mom,
            "mom_bwd_loss": mom_bwd,
            "sup_loss": sup,
            "wasserstein_gap": w_gap,
        }

    return train_step


def train_latent_cyclegan(
    train_src: np.ndarray,
    train_tgt: np.ndarray,
    source_name: str,
    target_name: str,
    epochs: int | None = None,
    batch_size: int | None = None,
    n_critic: int | None = None,
    config: CycleGANTrainingConfig | Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Train the shared latent CycleGAN using an explicit run configuration.

    ``epochs``, ``batch_size`` and ``n_critic`` remain accepted for backward
    compatibility with the neuronal application. When ``config`` is omitted,
    behavior matches the existing neuronal defaults.
    """
    if config is None:
        cfg = CycleGANTrainingConfig()
    elif isinstance(config, dict):
        cfg = CycleGANTrainingConfig.from_dict(config)
    else:
        cfg = config
    if epochs is not None:
        cfg = replace(cfg, epochs=int(epochs))
    if batch_size is not None:
        cfg = replace(cfg, batch_size=int(batch_size))
    if n_critic is not None:
        cfg = replace(cfg, n_critic=int(n_critic))

    tf.keras.backend.clear_session()
    # Original perturbation GAN runs did not set a TensorFlow RNG seed. Set
    # ``tf_seed: null`` in those archived-run configs to preserve that behavior.
    if cfg.tf_seed is not None:
        set_tf_seed(int(cfg.tf_seed))

    train_src = np.asarray(train_src, dtype=np.float32)
    train_tgt = np.asarray(train_tgt, dtype=np.float32)
    if train_src.ndim != 2 or train_tgt.ndim != 2:
        raise ValueError("CycleGAN inputs must both be 2-D latent matrices.")
    if not np.isfinite(train_src).all() or not np.isfinite(train_tgt).all():
        raise ValueError("CycleGAN latent inputs contain NaN/Inf values.")
    if train_src.shape[1] != train_tgt.shape[1]:
        raise ValueError(f"Latent dimension mismatch: {train_src.shape[1]} vs {train_tgt.shape[1]}")
    if min(train_src.shape[0], train_tgt.shape[0]) < cfg.k_pair:
        raise ValueError("Target/source training set is smaller than k_pair.")

    if cfg.steps_per_epoch_from == "source":
        steps_per_epoch = len(train_src) // cfg.batch_size
    elif cfg.steps_per_epoch_from == "target":
        steps_per_epoch = len(train_tgt) // cfg.batch_size
    else:
        raise ValueError("steps_per_epoch_from must be 'source' or 'target'.")
    if steps_per_epoch < 1:
        raise ValueError("Training set is smaller than the configured CycleGAN batch size.")

    shuffle_kwargs_src = {"buffer_size": cfg.source_shuffle_buffer}
    shuffle_kwargs_tgt = {"buffer_size": cfg.target_shuffle_buffer}
    if cfg.tf_seed is not None:
        shuffle_kwargs_src.update(seed=int(cfg.tf_seed), reshuffle_each_iteration=True)
        shuffle_kwargs_tgt.update(seed=int(cfg.tf_seed), reshuffle_each_iteration=True)

    ds_src = tf.data.Dataset.from_tensor_slices(train_src).shuffle(**shuffle_kwargs_src).repeat()
    ds_tgt = tf.data.Dataset.from_tensor_slices(train_tgt).shuffle(**shuffle_kwargs_tgt).repeat()
    train_dataset = (
        tf.data.Dataset.zip((ds_src, ds_tgt))
        .batch(cfg.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    tgt_bank = tf.constant(train_tgt, dtype=tf.float32)

    input_dim = train_src.shape[1]
    output_dim = train_tgt.shape[1]
    G_fwd = build_generator_with_affine_head(
        input_dim, output_dim, name_prefix=f"G_{source_name}_to_{target_name}"
    )
    G_bwd = build_generator(output_dim, input_dim, name=f"G_{target_name}_to_{source_name}")
    D_tgt = build_discriminator(output_dim, name=f"D_{target_name}")
    D_src = build_discriminator(input_dim, name=f"D_{source_name}")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg.lr_initial,
        decay_steps=cfg.lr_decay_steps,
        decay_rate=cfg.lr_decay_rate,
        staircase=True,
    )
    opt_G_fwd = Adam(learning_rate=lr_schedule, beta_1=cfg.adam_beta1, clipnorm=cfg.clipnorm)
    opt_G_bwd = Adam(learning_rate=lr_schedule, beta_1=cfg.adam_beta1, clipnorm=cfg.clipnorm)
    opt_D_tgt = Adam(learning_rate=lr_schedule, beta_1=cfg.adam_beta1, clipnorm=cfg.clipnorm)
    opt_D_src = Adam(learning_rate=lr_schedule, beta_1=cfg.adam_beta1, clipnorm=cfg.clipnorm)

    train_step_fn = make_cyclegan_train_step(
        G_fwd,
        G_bwd,
        D_tgt,
        D_src,
        opt_G_fwd,
        opt_G_bwd,
        opt_D_tgt,
        opt_D_src,
        tgt_bank,
        config=cfg,
    )

    histories = {k: [] for k in HISTORY_KEYS}
    pair_src = tf.zeros([0, input_dim], tf.float32)
    pair_tgt = tf.zeros([0, output_dim], tf.float32)

    for epoch in range(1, cfg.epochs + 1):
        if (epoch % cfg.mnn_every) == 1 and cfg.lambda_sup > 0.0:
            n = min(cfg.mnn_pool, train_src.shape[0], train_tgt.shape[0])
            idx_src = np.random.choice(train_src.shape[0], n, replace=False)
            idx_tgt = np.random.choice(
                train_tgt.shape[0], n, replace=bool(cfg.mnn_target_replace)
            )
            z_src = train_src[idx_src]
            z_tgt = train_tgt[idx_tgt]
            z_fake = G_fwd(tf.constant(z_src, tf.float32), training=False).numpy()
            z_fake = np.nan_to_num(z_fake, nan=0.0, posinf=1e6, neginf=-1e6)
            z_tgt = np.nan_to_num(z_tgt, nan=0.0, posinf=1e6, neginf=-1e6)
            ii, jj = make_mnn_pairs(z_fake, z_tgt)
            if len(ii) > 0:
                pair_src = tf.constant(z_src[ii], tf.float32)
                pair_tgt = tf.constant(z_tgt[jj], tf.float32)
            else:
                pair_src = tf.zeros([0, input_dim], tf.float32)
                pair_tgt = tf.zeros([0, output_dim], tf.float32)

        ep = {k: 0.0 for k in histories}
        steps = 0
        for real_src, real_tgt in train_dataset.take(steps_per_epoch):
            out = train_step_fn(
                real_src, real_tgt, pair_src, pair_tgt, tf.constant(cfg.n_critic)
            )
            for key in ep:
                ep[key] += float(out[key])
            steps += 1
        for key in ep:
            histories[key].append(ep[key] / max(steps, 1))

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[{source_name}->{target_name}] Epoch {epoch:03d} | "
                f"W-gap {histories['wasserstein_gap'][-1]:.3f} | "
                f"gen_fwd {histories['gen_fwd_loss'][-1]:.3f} | "
                f"disc_tgt {histories['disc_tgt_loss'][-1]:.3f} | "
                f"pair {histories['pair_loss'][-1]:.3f} | "
                f"mmd {histories['mmd_loss'][-1]:.3f} | "
                f"cycle {histories['cycle_loss'][-1]:.3f} | "
                f"id {histories['identity_loss'][-1]:.3f} | "
                f"mom {histories['mom_loss'][-1]:.3f} | "
                f"mom_bwd {histories['mom_bwd_loss'][-1]:.3f}"
            )

    return {
        "G_fwd": G_fwd,
        "G_bwd": G_bwd,
        "D_tgt": D_tgt,
        "D_src": D_src,
        "histories": histories,
        "source_name": source_name,
        "target_name": target_name,
        "steps_per_epoch": int(steps_per_epoch),
        "training_config": asdict(cfg),
    }


def plot_gan_training_curves(
    histories: Dict[str, list],
    title_prefix: str,
    logy: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot the major CycleGAN training losses and optionally save the figure."""
    epochs = np.arange(1, len(histories["gen_fwd_loss"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(epochs, histories["gen_fwd_loss"], label="Generator (forward) loss", lw=2)
    ax.plot(epochs, histories["gen_bwd_loss"], label="Generator (backward) loss", lw=2)
    ax.plot(epochs, histories["disc_tgt_loss"], label="Discriminator (target) loss", lw=2)
    ax.plot(epochs, histories["disc_src_loss"], label="Discriminator (source) loss", lw=2)
    ax.plot(epochs, histories["cycle_loss"], label="Cycle loss", lw=2)
    ax.plot(epochs, histories["identity_loss"], label="Identity loss", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title_prefix} -- major training losses")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
