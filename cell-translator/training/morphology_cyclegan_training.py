"""CycleGAN training routine for Cell Painting morphology translation.

This module preserves the morphology-specific latent standardization, network
sizes, MMD bandwidths, loss weights, and learning-rate schedule used in the
reported Cell Painting analyses.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd

from cell_translator.models.morphology_cyclegan import (
    build_morphology_generator,
    build_morphology_generator_with_head,
    build_morphology_discriminator,
)
from cell_translator.training.morphology_ae_training import (
    load_state_artifacts,
    load_morphology_ae,
    decode_latents,
    inverse_scale,
)


@dataclass(frozen=True)
class MorphologyCycleGANConfig:
    epochs: int = 300
    batch_size: int = 32
    gp_weight: float = 10.0
    drift_eps: float = 1e-3
    n_critic: int = 5
    lambda_cycle: float = 5.0
    lambda_identity: float = 0.5
    pair_weight: float = 1.0
    mmd_weight: float = 1.0
    covariance_weight: float = 0.2
    moment_weight: float = 2.0
    knn_k: int = 5
    bank_sample: int = 8192
    mmd_sigmas: tuple[float, ...] = (0.5, 1.0, 2.0)
    learning_rate: float = 1e-4
    lr_decay_steps: int = 8000
    lr_decay_rate: float = 0.9
    adam_beta1: float = 0.5
    clipnorm: float = 1.0
    shuffle_buffer: int = 4096
    steps_per_epoch_from: str = "minimum"
    # The source notebook did not explicitly seed TensorFlow.
    tensorflow_seed: int | None = None

    @classmethod
    def from_dict(cls, values: dict | None):
        values = dict(values or {})
        if "mmd_sigmas" in values:
            values["mmd_sigmas"] = tuple(float(x) for x in values["mmd_sigmas"])
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in values.items() if k in allowed})


def _tf():
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    return tf, Adam


def generator_loss(fake_scores):
    tf, _ = _tf()
    return -tf.reduce_mean(fake_scores)


def discriminator_loss(real_scores, fake_scores):
    tf, _ = _tf()
    return tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)


def gradient_penalty(discriminator, real, fake):
    tf, _ = _tf()
    alpha = tf.random.uniform([tf.shape(real)[0], 1])
    inter = real + alpha * (fake - real)
    with tf.GradientTape() as tape:
        tape.watch(inter)
        pred = discriminator(inter)
    grad = tape.gradient(pred, inter)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1) + 1e-12)
    return tf.reduce_mean((norm - 1.0) ** 2)


def knn_pair_loss_cos(fake, bank, k=5, bank_sample=8192):
    tf, _ = _tf()
    n = tf.shape(bank)[0]
    m = tf.minimum(bank_sample, n)
    idx = tf.random.uniform([m], 0, n, dtype=tf.int32)
    B = tf.gather(bank, idx)
    f = tf.math.l2_normalize(fake, axis=1)
    b = tf.math.l2_normalize(B, axis=1)
    sims = tf.matmul(f, b, transpose_b=True)
    _, inds = tf.math.top_k(sims, k=k)
    b_knn = tf.gather(b, inds, axis=0)
    return tf.reduce_mean(1.0 - tf.reduce_sum(f * tf.reduce_mean(b_knn, axis=1), axis=1))


def mmd_rbf(x, y, sigmas=(0.5, 1.0, 2.0)):
    tf, _ = _tf()

    def kernel(a, b, sigma):
        a2 = tf.reduce_sum(a * a, 1, keepdims=True)
        b2 = tf.reduce_sum(b * b, 1, keepdims=True)
        d2 = a2 - 2.0 * tf.matmul(a, b, transpose_b=True) + tf.transpose(b2)
        return tf.exp(-d2 / (2.0 * sigma * sigma))

    kxx = tf.add_n([kernel(x, x, s) for s in sigmas]) / len(sigmas)
    kyy = tf.add_n([kernel(y, y, s) for s in sigmas]) / len(sigmas)
    kxy = tf.add_n([kernel(x, y, s) for s in sigmas]) / len(sigmas)
    return tf.reduce_mean(kxx) + tf.reduce_mean(kyy) - 2.0 * tf.reduce_mean(kxy)


def moment_loss(fake, real, eps=1e-6):
    """Historical morphology convention: epsilon is inside |std_fake-std_real+eps|."""
    tf, _ = _tf()
    return (
        tf.reduce_mean(tf.abs(tf.reduce_mean(fake, 0) - tf.reduce_mean(real, 0)))
        + tf.reduce_mean(tf.abs(tf.math.reduce_std(fake, 0) - tf.math.reduce_std(real, 0) + eps))
    )


def covariance_loss(fake, real):
    tf, _ = _tf()

    def covariance(x):
        xc = x - tf.reduce_mean(x, 0, keepdims=True)
        return tf.matmul(xc, xc, transpose_a=True) / tf.cast(tf.shape(xc)[0] - 1, tf.float32)

    return tf.reduce_mean(tf.square(covariance(fake) - covariance(real)))


def _steps_per_epoch(n_src, n_tgt, batch_size, mode):
    mode = str(mode).lower()
    if mode == "source":
        n = n_src
    elif mode == "target":
        n = n_tgt
    elif mode == "minimum":
        n = min(n_src, n_tgt)
    else:
        raise ValueError("steps_per_epoch_from must be 'source', 'target', or 'minimum'.")
    return max(int(n) // int(batch_size), 1)


def train_morphology_cyclegan(source_state, target_state, artifact_dir, output_dir, config):
    """Train a morphology CycleGAN and decode held-out source cells to target features."""
    tf, Adam = _tf()
    tf.keras.backend.clear_session()
    if config.tensorflow_seed is not None:
        tf.keras.utils.set_random_seed(int(config.tensorflow_seed))

    source = load_state_artifacts(source_state, artifact_dir)
    target = load_state_artifacts(target_state, artifact_dir)

    mean_src = source["Z_train"].mean(0, keepdims=True).astype(np.float32)
    std_src = np.maximum(source["Z_train"].std(0, keepdims=True), 1e-6).astype(np.float32)
    mean_tgt = target["Z_train"].mean(0, keepdims=True).astype(np.float32)
    std_tgt = np.maximum(target["Z_train"].std(0, keepdims=True), 1e-6).astype(np.float32)

    train_src = ((source["Z_train"] - mean_src) / std_src).astype(np.float32)
    train_tgt = ((target["Z_train"] - mean_tgt) / std_tgt).astype(np.float32)
    validation_src = ((source["Z_validation"] - mean_src) / std_src).astype(np.float32)

    if train_src.shape[1] != train_tgt.shape[1]:
        raise ValueError("Source and target latent dimensions must match.")

    dim = train_src.shape[1]
    target_bank = tf.constant(train_tgt, dtype=tf.float32)
    steps = _steps_per_epoch(
        len(train_src), len(train_tgt), config.batch_size, config.steps_per_epoch_from
    )

    ds_src = tf.data.Dataset.from_tensor_slices(train_src).shuffle(config.shuffle_buffer).repeat()
    ds_tgt = tf.data.Dataset.from_tensor_slices(train_tgt).shuffle(config.shuffle_buffer).repeat()
    dataset = (
        tf.data.Dataset.zip((ds_src, ds_tgt))
        .batch(config.batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    safe_src = source_state.replace(" ", "_")
    safe_tgt = target_state.replace(" ", "_")
    G_fwd = build_morphology_generator_with_head(dim, dim, f"G_{safe_src}_to_{safe_tgt}")
    G_bwd = build_morphology_generator(dim, dim, f"G_{safe_tgt}_to_{safe_src}")
    D_tgt = build_morphology_discriminator(dim, f"D_{safe_tgt}")
    D_src = build_morphology_discriminator(dim, f"D_{safe_src}")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        config.learning_rate,
        config.lr_decay_steps,
        config.lr_decay_rate,
        staircase=True,
    )
    opt_fwd = Adam(lr_schedule, beta_1=config.adam_beta1, clipnorm=config.clipnorm)
    opt_bwd = Adam(lr_schedule, beta_1=config.adam_beta1, clipnorm=config.clipnorm)
    opt_D_tgt = Adam(lr_schedule, beta_1=config.adam_beta1, clipnorm=config.clipnorm)
    opt_D_src = Adam(lr_schedule, beta_1=config.adam_beta1, clipnorm=config.clipnorm)

    @tf.function
    def train_step(real_src, real_tgt):
        for _ in tf.range(config.n_critic):
            with tf.GradientTape(persistent=True) as tape_d:
                fake_tgt = G_fwd(real_src, training=True)
                fake_src = G_bwd(real_tgt, training=True)
                r_tgt = D_tgt(real_tgt, training=True)
                f_tgt = D_tgt(fake_tgt, training=True)
                r_src = D_src(real_src, training=True)
                f_src = D_src(fake_src, training=True)
                d_tgt = (
                    discriminator_loss(r_tgt, f_tgt)
                    + config.gp_weight * gradient_penalty(D_tgt, real_tgt, fake_tgt)
                    + config.drift_eps * (tf.reduce_mean(r_tgt ** 2) + tf.reduce_mean(f_tgt ** 2))
                )
                d_src = (
                    discriminator_loss(r_src, f_src)
                    + config.gp_weight * gradient_penalty(D_src, real_src, fake_src)
                    + config.drift_eps * (tf.reduce_mean(r_src ** 2) + tf.reduce_mean(f_src ** 2))
                )
            opt_D_tgt.apply_gradients(zip(tape_d.gradient(d_tgt, D_tgt.trainable_variables), D_tgt.trainable_variables))
            opt_D_src.apply_gradients(zip(tape_d.gradient(d_src, D_src.trainable_variables), D_src.trainable_variables))
            del tape_d

        w_gap = tf.reduce_mean(r_tgt) - tf.reduce_mean(f_tgt)

        with tf.GradientTape(persistent=True) as tape_g:
            fake_tgt = G_fwd(real_src, training=True)
            fake_src = G_bwd(real_tgt, training=True)
            cyc_src = tf.reduce_mean(tf.abs(real_src - G_bwd(fake_tgt, training=True)))
            cyc_tgt = tf.reduce_mean(tf.abs(real_tgt - G_fwd(fake_src, training=True)))
            id_fwd = tf.reduce_mean(tf.abs(real_tgt - G_fwd(real_tgt, training=True)))
            id_bwd = tf.reduce_mean(tf.abs(real_src - G_bwd(real_src, training=True)))
            pair = knn_pair_loss_cos(fake_tgt, target_bank, config.knn_k, config.bank_sample)
            mmd = mmd_rbf(fake_tgt, real_tgt, config.mmd_sigmas)
            cov = covariance_loss(fake_tgt, real_tgt)
            mom = moment_loss(fake_tgt, real_tgt)
            mom_bwd = moment_loss(fake_src, real_src)
            loss_fwd = (
                generator_loss(D_tgt(fake_tgt, training=True))
                + config.lambda_cycle * (cyc_src + cyc_tgt)
                + config.lambda_identity * id_fwd
                + config.pair_weight * pair
                + config.mmd_weight * mmd
                + config.covariance_weight * cov
                + config.moment_weight * mom
            )
            loss_bwd = (
                generator_loss(D_src(fake_src, training=True))
                + config.lambda_cycle * (cyc_src + cyc_tgt)
                + config.lambda_identity * id_bwd
                + config.moment_weight * mom_bwd
            )
        opt_fwd.apply_gradients(zip(tape_g.gradient(loss_fwd, G_fwd.trainable_variables), G_fwd.trainable_variables))
        opt_bwd.apply_gradients(zip(tape_g.gradient(loss_bwd, G_bwd.trainable_variables), G_bwd.trainable_variables))
        del tape_g

        return {
            "gen_fwd_loss": loss_fwd,
            "gen_bwd_loss": loss_bwd,
            "disc_tgt_loss": d_tgt,
            "disc_src_loss": d_src,
            "cycle_loss": cyc_src + cyc_tgt,
            "identity_loss": id_fwd + id_bwd,
            "pair_loss": pair,
            "mmd_loss": mmd,
            "cov_loss": cov,
            "moment_loss": mom,
            "moment_bwd_loss": mom_bwd,
            "wasserstein_gap": w_gap,
        }

    keys = [
        "gen_fwd_loss", "gen_bwd_loss", "disc_tgt_loss", "disc_src_loss",
        "cycle_loss", "identity_loss", "pair_loss", "mmd_loss", "cov_loss",
        "moment_loss", "moment_bwd_loss", "wasserstein_gap",
    ]
    history = {key: [] for key in keys}

    for epoch in range(1, config.epochs + 1):
        totals = {key: 0.0 for key in keys}
        n_steps = 0
        for real_src, real_tgt in dataset.take(steps):
            values = train_step(real_src, real_tgt)
            for key in keys:
                totals[key] += float(values[key])
            n_steps += 1
        for key in keys:
            history[key].append(totals[key] / max(n_steps, 1))
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | W-gap={history['wasserstein_gap'][-1]:.3f} | "
                f"gen={history['gen_fwd_loss'][-1]:.3f} | mmd={history['mmd_loss'][-1]:.3f} | "
                f"cycle={history['cycle_loss'][-1]:.3f} | mom={history['moment_loss'][-1]:.3f}"
            )

    output = Path(output_dir)
    run_dir = output / f"{safe_src}_to_{safe_tgt}"
    run_dir.mkdir(parents=True, exist_ok=True)
    G_fwd.save(run_dir / "generator_forward.keras")
    G_bwd.save(run_dir / "generator_backward.keras")
    D_tgt.save(run_dir / "critic_target.keras")
    D_src.save(run_dir / "critic_source.keras")
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    fake_tgt_std = G_fwd(tf.constant(validation_src, dtype=tf.float32), training=False).numpy()
    fake_tgt_latent = fake_tgt_std * std_tgt + mean_tgt
    target_model, _ = load_morphology_ae(target_state, artifact_dir)
    translated_scaled = decode_latents(
        target_model, fake_tgt_latent, next(target_model.parameters()).device
    )
    translated_original = inverse_scale(translated_scaled, target["scaler_stats"])
    target_validation_original = inverse_scale(target["X_validation_scaled"], target["scaler_stats"])
    source_validation_original = inverse_scale(source["X_validation_scaled"], source["scaler_stats"])

    np.save(run_dir / "translated_features.npy", translated_original)
    np.save(run_dir / "translated_latent.npy", fake_tgt_latent.astype(np.float32))
    np.save(run_dir / "source_validation_features.npy", source_validation_original)
    np.save(run_dir / "target_validation_features.npy", target_validation_original)

    with open(run_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_state": source_state,
                "target_state": target_state,
                "steps_per_epoch": steps,
                "config": asdict(config),
            },
            handle,
            indent=2,
        )

    return {
        "run_dir": str(run_dir),
        "translated_features": translated_original,
        "source_validation_features": source_validation_original,
        "target_validation_features": target_validation_original,
        "history": history,
        "G_fwd": G_fwd,
        "G_bwd": G_bwd,
    }
