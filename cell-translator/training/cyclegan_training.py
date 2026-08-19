"""Latent-space WGAN-GP CycleGAN losses and training routine.

The implementation is shared across Cell Translator applications and
preserves the model architecture, loss functions, and training procedure.
"""

import os
from typing import Dict, Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from cell_translator.hyperparameters import *
from cell_translator.models.cyclegan import (
    build_generator, build_discriminator, build_generator_with_affine_head,
)
from cell_translator.utils.reproducibility import set_tf_seed

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
    with tf.GradientTape() as t:
        t.watch(inter)
        pred = discriminator(inter)
    grad = t.gradient(pred, inter)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1) + 1e-12)
    return tf.reduce_mean((norm - 1.0) ** 2)

def knn_pair_loss_cos(fake_tgt, tgt_bank, k=5, bank_sample=8192):
    """Cosine k-NN pull: pushes fake target latents toward the real target manifold."""
    n_bank = tf.shape(tgt_bank)[0]
    m = tf.minimum(bank_sample, n_bank)
    idx = tf.random.uniform([m], 0, n_bank, dtype=tf.int32)
    B = tf.gather(tgt_bank, idx)                          # (M, d)
    f = tf.math.l2_normalize(fake_tgt, axis=1)            # cosine space
    b = tf.math.l2_normalize(B, axis=1)
    sims = tf.matmul(f, b, transpose_b=True)              # (batch, M)
    _, inds = tf.math.top_k(sims, k=k)
    b_knn = tf.gather(b, inds, axis=0)                    # (batch, k, d)
    b_tgt = tf.reduce_mean(b_knn, axis=1)                 # (batch, d)
    return tf.reduce_mean(1.0 - tf.reduce_sum(f * b_tgt, axis=1))

def cov_loss(fake, real):
    """Frobenius covariance alignment."""
    def cov(x):
        xc = x - tf.reduce_mean(x, axis=0, keepdims=True)
        n = tf.cast(tf.shape(xc)[0] - 1, tf.float32)
        return tf.matmul(xc, xc, transpose_a=True) / n
    return tf.reduce_mean(tf.square(cov(fake) - cov(real)))

def moment_loss(fake, real, eps=1e-6):
    """Mean + std matching per latent dimension."""
    mf = tf.reduce_mean(fake, axis=0)
    mr = tf.reduce_mean(real, axis=0)
    sf = tf.math.reduce_std(fake, axis=0) + eps
    sr = tf.math.reduce_std(real, axis=0) + eps
    return tf.reduce_mean(tf.abs(mf - mr)) + tf.reduce_mean(tf.abs(sf - sr))

def mmd_rbf(x, y, sigmas=(1.0, 2.0, 4.0)):
    """Multi-scale RBF MMD."""
    def k(a, b, s):
        g = 1.0 / (2 * s * s)
        a2 = tf.reduce_sum(a * a, 1, keepdims=True)
        b2 = tf.reduce_sum(b * b, 1, keepdims=True)
        d2 = a2 - 2.0 * tf.matmul(a, b, transpose_b=True) + tf.transpose(b2)
        return tf.exp(-g * d2)
    kxx = tf.add_n([k(x, x, s) for s in sigmas]) / len(sigmas)
    kyy = tf.add_n([k(y, y, s) for s in sigmas]) / len(sigmas)
    kxy = tf.add_n([k(x, y, s) for s in sigmas]) / len(sigmas)
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

    I = np.array([i for i, _ in mnn], dtype=int)
    J = np.array([j for _, j in mnn], dtype=int)
    return I, J

def make_cyclegan_train_step(
    G_fwd, G_bwd, D_tgt, D_src,
    opt_G_fwd, opt_G_bwd, opt_D_tgt, opt_D_src,
    tgt_bank,
):
    """
    Build the @tf.function train step shared by all three translation
    directions. This is a straight factory-function extraction of the
    notebook's train_step (identical across the iPSC->NPC, NPC->Neuron and
    NPC->Astrocyte cells, save for source/target variable names) -- every
    operation, loss term and weight below is unchanged from the original.
    """

    @tf.function
    def train_step(real_src, real_tgt, pair_src, pair_tgt, n_critic=tf.constant(N_CRITIC)):
        # ----- train discriminators -----
        for _ in tf.range(n_critic):
            with tf.GradientTape(persistent=True) as tD:
                fake_tgt = G_fwd(real_src, training=True)
                fake_src = G_bwd(real_tgt, training=True)

                r_tgt = D_tgt(real_tgt, training=True); f_tgt = D_tgt(fake_tgt, training=True)
                r_src = D_src(real_src, training=True); f_src = D_src(fake_src, training=True)

                gp_tgt = gradient_penalty(D_tgt, real_tgt, fake_tgt)
                gp_src = gradient_penalty(D_src, real_src, fake_src)

                drift_tgt = DRIFT_EPS * (tf.reduce_mean(tf.square(r_tgt)) + tf.reduce_mean(tf.square(f_tgt)))
                drift_src = DRIFT_EPS * (tf.reduce_mean(tf.square(r_src)) + tf.reduce_mean(tf.square(f_src)))

                disc_tgt_loss = discriminator_loss(r_tgt, f_tgt) + GP_WEIGHT * gp_tgt + drift_tgt
                disc_src_loss = discriminator_loss(r_src, f_src) + GP_WEIGHT * gp_src + drift_src

            d_tgt_grads = tD.gradient(disc_tgt_loss, D_tgt.trainable_variables)
            d_src_grads = tD.gradient(disc_src_loss, D_src.trainable_variables)
            opt_D_tgt.apply_gradients(zip(d_tgt_grads, D_tgt.trainable_variables))
            opt_D_src.apply_gradients(zip(d_src_grads, D_src.trainable_variables))
            del tD

        # Wasserstein gap (from last critic iteration)
        w_gap = tf.reduce_mean(r_tgt) - tf.reduce_mean(f_tgt)

        # ----- train generators -----
        with tf.GradientTape(persistent=True) as tape_g:
            fake_tgt = G_fwd(real_src, training=True)
            fake_src = G_bwd(real_tgt, training=True)

            # adversarial
            gen_fwd_adv = generator_loss(D_tgt(fake_tgt, training=True))
            gen_bwd_adv = generator_loss(D_src(fake_src, training=True))

            # cycle consistency
            cycled_src = G_bwd(fake_tgt, training=True)
            cycled_tgt = G_fwd(fake_src, training=True)
            cyc_src = cycle_loss(real_src, cycled_src)
            cyc_tgt = cycle_loss(real_tgt, cycled_tgt)

            # identity
            id_tgt = G_fwd(real_tgt, training=True)
            id_src = G_bwd(real_src, training=True)
            id_loss_fwd = identity_loss(real_tgt, id_tgt)
            id_loss_bwd = identity_loss(real_src, id_src)

            # distribution matching losses (forward direction only)
            pair = knn_pair_loss_cos(fake_tgt, tgt_bank, k=K_PAIR, bank_sample=BANK_SAMPLE)
            mmd = mmd_rbf(fake_tgt, real_tgt)
            cval = cov_loss(fake_tgt, real_tgt)
            mom = moment_loss(fake_tgt, real_tgt)

            # moment loss on the backward generator, to strengthen cycle consistency
            mom_bwd = moment_loss(fake_src, real_src)

            # supervised MNN term (safe even when LAMBDA_SUP=0)
            def sup_term():
                pred = G_fwd(pair_src, training=True)
                return tf.reduce_mean(tf.square(pred - pair_tgt))

            sup = tf.cond(
                tf.logical_and(LAMBDA_SUP > 0.0, tf.shape(pair_src)[0] > 0),
                sup_term,
                lambda: tf.constant(0.0, tf.float32),
            )

            # total losses
            total_gen_fwd_loss = (
                gen_fwd_adv
                + LAMBDA_CYCLE * (cyc_src + cyc_tgt)
                + LAMBDA_IDENTITY * id_loss_fwd
                + PAIR_W * pair
                + MMD_W * mmd
                + COV_W * cval
                + MOM_W * mom
                + LAMBDA_SUP * sup
            )

            total_gen_bwd_loss = (
                gen_bwd_adv
                + LAMBDA_CYCLE * (cyc_src + cyc_tgt)
                + LAMBDA_IDENTITY * id_loss_bwd
                + MOM_W * mom_bwd
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
    epochs: int = CGAN_EPOCHS,
    batch_size: int = CGAN_BATCH_SIZE,
    n_critic: int = N_CRITIC,
) -> Dict[str, Any]:
    """
    Shared CycleGAN training routine operating on standardized HurdleAE
    latent vectors. `train_src`/`train_tgt` are (n_cells, latent_dim)
    float32 arrays for the two states being translated between.

    This single function is used by all three translation directions
    (translate_ipsc_to_npc, translate_npc_to_neuron,
    translate_npc_to_astrocyte) -- the architecture, loss functions, loss
    weights, optimizer settings and training loop are identical across
    directions; only the input latents, names, and the resulting
    save/plot labels differ.
    """
    tf.keras.backend.clear_session()  # avoid layer/model name collisions across repeated calls
    set_tf_seed(SEED)

    train_src = np.asarray(train_src, dtype=np.float32)
    train_tgt = np.asarray(train_tgt, dtype=np.float32)
    if train_src.ndim != 2 or train_tgt.ndim != 2:
        raise ValueError("CycleGAN inputs must both be 2-D latent matrices.")
    if not np.isfinite(train_src).all() or not np.isfinite(train_tgt).all():
        raise ValueError("CycleGAN latent inputs contain NaN/Inf values.")

    input_dim = train_src.shape[1]
    output_dim = train_tgt.shape[1]
    assert input_dim == output_dim, f"Dim mismatch: {input_dim} vs {output_dim}"

    steps_per_epoch = len(train_tgt) // batch_size
    if steps_per_epoch < 1:
        raise ValueError(
            f"Target training set has {len(train_tgt)} cells, fewer than the "
            f"CycleGAN batch size ({batch_size})."
        )

    ds_src = tf.data.Dataset.from_tensor_slices(train_src).shuffle(
        4096, seed=SEED, reshuffle_each_iteration=True
    ).repeat()
    ds_tgt = tf.data.Dataset.from_tensor_slices(train_tgt).shuffle(
        4096, seed=SEED, reshuffle_each_iteration=True
    ).repeat()
    train_dataset = (
        tf.data.Dataset.zip((ds_src, ds_tgt))
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    tgt_bank = tf.constant(train_tgt, dtype=tf.float32)

    name_prefix = f"G_{source_name}_to_{target_name}"
    G_fwd = build_generator_with_affine_head(input_dim, output_dim, name_prefix=name_prefix)
    G_bwd = build_generator(output_dim, input_dim, name=f"G_{target_name}_to_{source_name}")
    D_tgt = build_discriminator(output_dim, name=f"D_{target_name}")
    D_src = build_discriminator(input_dim, name=f"D_{source_name}")

    print(f"G_fwd params: {G_fwd.count_params():,}")
    print(f"G_bwd params: {G_bwd.count_params():,}")
    print(f"D_tgt params: {D_tgt.count_params():,}")
    print(f"D_src params: {D_src.count_params():,}")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LR_INITIAL,
        decay_steps=LR_DECAY_STEPS,
        decay_rate=LR_DECAY_RATE,
        staircase=True,
    )
    opt_G_fwd = Adam(learning_rate=lr_schedule, beta_1=ADAM_BETA1, clipnorm=CLIPNORM)
    opt_G_bwd = Adam(learning_rate=lr_schedule, beta_1=ADAM_BETA1, clipnorm=CLIPNORM)
    opt_D_tgt = Adam(learning_rate=lr_schedule, beta_1=ADAM_BETA1, clipnorm=CLIPNORM)
    opt_D_src = Adam(learning_rate=lr_schedule, beta_1=ADAM_BETA1, clipnorm=CLIPNORM)

    train_step_fn = make_cyclegan_train_step(
        G_fwd, G_bwd, D_tgt, D_src,
        opt_G_fwd, opt_G_bwd, opt_D_tgt, opt_D_src,
        tgt_bank,
    )

    histories = {k: [] for k in HISTORY_KEYS}
    empty_src = tf.zeros([0, input_dim], tf.float32)
    empty_tgt = tf.zeros([0, input_dim], tf.float32)
    pair_src, pair_tgt = empty_src, empty_tgt

    for epoch in range(1, epochs + 1):
        # MNN pair refresh (only when LAMBDA_SUP > 0; disabled by default)
        if (epoch % MNN_EVERY) == 1 and LAMBDA_SUP > 0.0:
            n = min(MNN_POOL, train_src.shape[0], train_tgt.shape[0])
            idxS = np.random.choice(train_src.shape[0], n, replace=False)
            idxT = np.random.choice(train_tgt.shape[0], n, replace=False)
            Zs = train_src[idxS]
            Zt = train_tgt[idxT]

            Zf = G_fwd(tf.constant(Zs, tf.float32), training=False).numpy()
            Zf = np.nan_to_num(Zf, nan=0.0, posinf=1e6, neginf=-1e6)
            Zt = np.nan_to_num(Zt, nan=0.0, posinf=1e6, neginf=-1e6)

            I, J = make_mnn_pairs(Zf, Zt)
            if len(I) > 0:
                pair_src = tf.constant(Zs[I], tf.float32)
                pair_tgt = tf.constant(Zt[J], tf.float32)
                print(f"  MNN pairs found: {len(I)}")
            else:
                pair_src, pair_tgt = empty_src, empty_tgt
                print(f"  MNN pairs: none found at epoch {epoch}")

        ep = {k: 0.0 for k in histories}
        steps = 0
        for real_src, real_tgt in train_dataset.take(steps_per_epoch):
            out = train_step_fn(real_src, real_tgt, pair_src, pair_tgt, tf.constant(n_critic))
            for k in ep:
                ep[k] += float(out[k])
            steps += 1

        for k in ep:
            histories[k].append(ep[k] / max(steps, 1))

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
                f"mom_bwd {histories['mom_bwd_loss'][-1]:.3f} | "
                f"sup {histories['sup_loss'][-1]:.3f}"
            )

    return {
        "G_fwd": G_fwd, "G_bwd": G_bwd, "D_tgt": D_tgt, "D_src": D_src,
        "histories": histories,
        "source_name": source_name, "target_name": target_name,
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
        print(f"Saved training curves: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
