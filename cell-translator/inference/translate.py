"""Generic latent loading, translation, decoding, and artifact saving.

This module is application-agnostic: biological applications provide their own
state names, input paths, and output directories.
"""

import os
import json
import platform
import sys
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import tensorflow as tf

from cell_translator.hyperparameters import *
from cell_translator.models.hurdle_ae import HurdleAutoencoder
from cell_translator.training.hurdle_ae_training import hard_decode
from cell_translator.training.cyclegan_training import (
    train_latent_cyclegan, plot_gan_training_curves,
)

def load_state_pair_latents(
    npz_path: str,
    state_names: List[str],
    source_state: str,
    target_state: str,
    x_paths: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """
    Load HurdleAE latents and return standardized source/target splits.

    Current NPZ files store state slices explicitly. For backward
    compatibility with outputs created by the original script, this function
    can reconstruct slices from the input matrices and can read `test_idx` as
    the historical name of the HurdleAE validation subset.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"HurdleAE output file not found: {npz_path}. "
            "Run with --train-ae or provide existing AE outputs."
        )

    npz = np.load(npz_path, allow_pickle=False)
    Z_all = npz["Z_all"]
    train_idx = npz["train_idx"]
    val_idx = npz["val_idx"] if "val_idx" in npz.files else npz["test_idx"]
    mean_z_train = npz["mean_z_train"]
    std_z_train = npz["std_z_train"]

    if source_state not in state_names or target_state not in state_names:
        raise ValueError(
            f"Requested states {source_state}->{target_state} are not both "
            f"present in state_names={state_names}."
        )

    # Prefer saved state metadata so inference does not depend on re-reading
    # expression matrices merely to recover cell counts.
    if all(k in npz.files for k in ("state_names", "slice_starts", "slice_ends")):
        saved_names = [str(x) for x in npz["state_names"].tolist()]
        starts = npz["slice_starts"].astype(np.int64)
        ends = npz["slice_ends"].astype(np.int64)
        slices = {name: (int(s), int(e)) for name, s, e in zip(saved_names, starts, ends)}
    else:
        slices = {}
        start = 0
        for name in state_names:
            if not os.path.isfile(x_paths[name]):
                raise FileNotFoundError(
                    f"Cannot reconstruct legacy state slices; missing {x_paths[name]}"
                )
            n = int(np.load(x_paths[name], mmap_mode="r").shape[0])
            slices[name] = (start, start + n)
            start += n

    for state in (source_state, target_state):
        if state not in slices:
            raise KeyError(f"State '{state}' is missing from saved AE state metadata.")

    def state_split(state: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx_state = np.arange(*slices[state], dtype=np.int64)
        idx_train = np.intersect1d(idx_state, train_idx)
        idx_val = np.intersect1d(idx_state, val_idx)
        return Z_all[idx_train], Z_all[idx_val], Z_all[idx_state], idx_train, idx_val

    src_train, src_val, src_all, src_train_idx, src_val_idx = state_split(source_state)
    tgt_train, tgt_val, tgt_all, tgt_train_idx, tgt_val_idx = state_split(target_state)

    def standardize(z: np.ndarray) -> np.ndarray:
        return ((z - mean_z_train) / std_z_train).astype(np.float32)

    train_src_z = standardize(src_train)
    train_tgt_z = standardize(tgt_train)
    val_src_z = standardize(src_val)
    val_tgt_z = standardize(tgt_val)
    all_src_z = standardize(src_all)
    all_tgt_z = standardize(tgt_all)

    print(f"{source_state} train: {train_src_z.shape} | validation: {val_src_z.shape}")
    print(f"{target_state} train: {train_tgt_z.shape} | validation: {val_tgt_z.shape}")

    return {
        "train_src": train_src_z,
        "train_tgt": train_tgt_z,
        "val_src": val_src_z,
        "val_tgt": val_tgt_z,
        "all_src": all_src_z,
        "all_tgt": all_tgt_z,
        "source_indices": np.arange(*slices[source_state], dtype=np.int64),
        "target_indices": np.arange(*slices[target_state], dtype=np.int64),
        "train_src_indices": src_train_idx.astype(np.int64),
        "val_src_indices": src_val_idx.astype(np.int64),
        "train_tgt_indices": tgt_train_idx.astype(np.int64),
        "val_tgt_indices": tgt_val_idx.astype(np.int64),
        "mean_z_train": mean_z_train.astype(np.float32),
        "std_z_train": std_z_train.astype(np.float32),
        # Backward-compatible aliases for code written against the original file.
        "test_src": val_src_z,
        "test_tgt": val_tgt_z,
    }

def _translation_slug(source_name: str, target_name: str) -> str:
    return f"{source_name.lower()}_to_{target_name.lower()}".replace(" ", "_")

def _ae_prefix_from_outputs(npz_path: str) -> str:
    suffix = "_outputs.npz"
    if not npz_path.endswith(suffix):
        raise ValueError(
            f"Expected an AE output path ending in '{suffix}', received: {npz_path}"
        )
    return npz_path[:-len(suffix)]

def _load_hurdle_ae_checkpoint(
    checkpoint_path: str,
    input_dim: int,
    device: torch.device,
) -> Tuple[HurdleAutoencoder, Dict[str, Any]]:
    """Load a trusted HurdleAE checkpoint created by this script."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"HurdleAE checkpoint not found: {checkpoint_path}")

    # PyTorch >=2.6 defaults torch.load to weights_only=True. This checkpoint
    # intentionally contains configuration/history metadata as well as weights.
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # compatibility with older PyTorch versions
        ckpt = torch.load(checkpoint_path, map_location=device)

    cfg = ckpt["config"]
    latent_dim = int(cfg["latent_dim"])
    dropout = float(cfg["dropout"])
    model = HurdleAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt

def decode_latents_with_hurdle_ae(
    model: HurdleAutoencoder,
    Z: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode unstandardized HurdleAE latents into detection P and magnitude MAG."""
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim != 2 or not np.isfinite(Z).all():
        raise ValueError("Latents to decode must be a finite 2-D array.")

    Ps, MAGs = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, Z.shape[0], batch_size):
            z = torch.from_numpy(Z[start:start + batch_size]).to(device)
            _, _, p, mag = model.decode(z)
            Ps.append(p.cpu().numpy())
            MAGs.append(mag.cpu().numpy())

    P = np.concatenate(Ps, axis=0).astype(np.float32)
    MAG = np.concatenate(MAGs, axis=0).astype(np.float32)
    return P, MAG

def save_cyclegan_artifacts(
    result: Dict[str, Any],
    save_dir: str,
    source_name: str,
    target_name: str,
) -> Dict[str, str]:
    """Save trained CycleGAN models, loss history, and exact run metadata."""
    out_dir = os.path.join(save_dir, _translation_slug(source_name, target_name))
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        "G_forward": os.path.join(out_dir, "G_forward.keras"),
        "G_backward": os.path.join(out_dir, "G_backward.keras"),
        "D_target": os.path.join(out_dir, "D_target.keras"),
        "D_source": os.path.join(out_dir, "D_source.keras"),
        "history": os.path.join(out_dir, "training_history.npz"),
        "curves": os.path.join(out_dir, "training_curves.png"),
        "translated": os.path.join(out_dir, "translated_source_outputs.npz"),
        "metadata": os.path.join(out_dir, "run_metadata.json"),
    }

    result["G_fwd"].save(paths["G_forward"])
    result["G_bwd"].save(paths["G_backward"])
    result["D_tgt"].save(paths["D_target"])
    result["D_src"].save(paths["D_source"])
    np.savez_compressed(
        paths["history"],
        **{k: np.asarray(v, dtype=np.float32) for k, v in result["histories"].items()},
    )

    metadata = {
        "source_state": source_name,
        "target_state": target_name,
        "seed": SEED,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pytorch": torch.__version__,
        "tensorflow": tf.__version__,
        "cyclegan_hyperparameters": result.get("training_config", {}),
    }
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved CycleGAN artifacts: {out_dir}")
    return paths

def translate_and_decode_source(
    result: Dict[str, Any],
    latents: Dict[str, np.ndarray],
    ae_outputs_npz: str,
    x_paths: Dict[str, str],
    source_state: str,
    target_state: str,
    output_path: str,
    split: str = "all",
) -> str:
    """
    Apply the trained forward generator to the requested source split and decode
    translated latent vectors back to target-state gene-expression space.

    Translation occurs in standardized latent space. Before HurdleAE decoding,
    translated vectors are returned to the original latent scale using the
    training-latent mean/std saved by Stage 1. Target-state `tau_gene` values
    are then used for the same hard-decoding rule as in HurdleAE reconstruction.
    """
    split_map = {
        "all": ("all_src", "source_indices"),
        "train": ("train_src", "train_src_indices"),
        "validation": ("val_src", "val_src_indices"),
        "val": ("val_src", "val_src_indices"),
        "test": ("val_src", "val_src_indices"),  # historical notebook terminology
    }
    if split not in split_map:
        raise ValueError(f"Unknown translation split '{split}'. Use all/train/validation.")
    latent_key, index_key = split_map[split]
    translated_z_std = result["G_fwd"](
        tf.constant(latents[latent_key], dtype=tf.float32), training=False
    ).numpy().astype(np.float32)

    mean_z = latents["mean_z_train"]
    std_z = latents["std_z_train"]
    translated_z = (translated_z_std * std_z + mean_z).astype(np.float32)

    ae_prefix = _ae_prefix_from_outputs(ae_outputs_npz)
    checkpoint_path = ae_prefix + "_best.pt"
    tau_path = ae_prefix + f"_tau_gene_{target_state}.npy"
    if not os.path.isfile(tau_path):
        raise FileNotFoundError(f"Target-state hard-decode threshold not found: {tau_path}")

    input_dim = int(np.load(x_paths[target_state], mmap_mode="r").shape[1])
    # Decode on CPU to avoid TensorFlow/PyTorch GPU-memory contention after
    # CycleGAN training. This affects execution location only, not the model.
    device = torch.device("cpu")
    ae_model, _ = _load_hurdle_ae_checkpoint(checkpoint_path, input_dim, device)
    P_translated, MAG_translated = decode_latents_with_hurdle_ae(
        ae_model, translated_z, device=device, batch_size=256
    )
    tau_target = np.load(tau_path).astype(np.float32)
    X_translated = hard_decode(P_translated, MAG_translated, tau_target)

    np.savez_compressed(
        output_path,
        source_state=np.asarray(source_state),
        target_state=np.asarray(target_state),
        source_indices=latents[index_key].astype(np.int64),
        translated_split=np.asarray(split),
        translated_z_standardized=translated_z_std,
        translated_z=translated_z,
        P_translated=P_translated,
        MAG_translated=MAG_translated,
        X_translated=X_translated,
        target_tau_gene=tau_target,
    )
    print(f"Saved translated source outputs: {output_path}")
    return output_path

def _run_translation(
    ae_outputs_npz: str,
    x_paths: Dict[str, str],
    state_names: List[str],
    source_state: str,
    target_state: str,
    save_dir: str,
    plot: bool,
    title_prefix: str,
    cyclegan_config: Dict[str, Any] | None = None,
    translation_split: str = "all",
) -> Dict[str, Any]:
    """Shared orchestration for one translation direction."""
    latents = load_state_pair_latents(
        ae_outputs_npz, state_names, source_state, target_state, x_paths
    )
    result = train_latent_cyclegan(
        latents["train_src"],
        latents["train_tgt"],
        source_name=source_state,
        target_name=target_state,
        config=cyclegan_config,
    )
    paths = save_cyclegan_artifacts(
        result, save_dir, source_name=source_state, target_name=target_state
    )
    plot_gan_training_curves(
        result["histories"],
        title_prefix=title_prefix,
        save_path=paths["curves"],
        show=plot,
    )
    translate_and_decode_source(
        result=result,
        latents=latents,
        ae_outputs_npz=ae_outputs_npz,
        x_paths=x_paths,
        source_state=source_state,
        target_state=target_state,
        output_path=paths["translated"],
        split=translation_split,
    )
    result["artifact_paths"] = paths
    return result


def run_translation(
    ae_outputs_npz: str,
    x_paths: Dict[str, str],
    state_names: List[str],
    source_state: str,
    target_state: str,
    save_dir: str,
    plot: bool = True,
    title_prefix: str | None = None,
    cyclegan_config: Dict[str, Any] | None = None,
    translation_split: str = "all",
) -> Dict[str, Any]:
    """Public application-agnostic wrapper around the preserved translation routine."""
    if title_prefix is None:
        title_prefix = f"{source_state} --> {target_state} latent translation"
    return _run_translation(
        ae_outputs_npz=ae_outputs_npz,
        x_paths=x_paths,
        state_names=state_names,
        source_state=source_state,
        target_state=target_state,
        save_dir=save_dir,
        plot=plot,
        title_prefix=title_prefix,
        cyclegan_config=cyclegan_config,
        translation_split=translation_split,
    )
