"""Inference helpers for archived Cell Painting morphology translators."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from cell_translator.training.morphology_ae_training import (
    load_state_artifacts,
    load_morphology_ae,
    decode_latents,
    inverse_scale,
)


def translate_with_archived_generator(
    source_state: str,
    target_state: str,
    artifact_dir: str,
    generator_path: str,
    output_path: str | None = None,
):
    """Translate held-out source cells with an archived forward Keras generator."""
    import tensorflow as tf

    source = load_state_artifacts(source_state, artifact_dir)
    target = load_state_artifacts(target_state, artifact_dir)
    mean_src = source["Z_train"].mean(0, keepdims=True).astype(np.float32)
    std_src = np.maximum(source["Z_train"].std(0, keepdims=True), 1e-6).astype(np.float32)
    mean_tgt = target["Z_train"].mean(0, keepdims=True).astype(np.float32)
    std_tgt = np.maximum(target["Z_train"].std(0, keepdims=True), 1e-6).astype(np.float32)

    source_validation_std = ((source["Z_validation"] - mean_src) / std_src).astype(np.float32)
    generator = tf.keras.models.load_model(generator_path, compile=False)
    translated_std = generator(
        tf.constant(source_validation_std, dtype=tf.float32), training=False
    ).numpy()
    translated_latent = translated_std * std_tgt + mean_tgt

    target_model, _ = load_morphology_ae(target_state, artifact_dir)
    translated_scaled = decode_latents(
        target_model, translated_latent, next(target_model.parameters()).device
    )
    translated_original = inverse_scale(translated_scaled, target["scaler_stats"])

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, translated_original)
    return translated_original
