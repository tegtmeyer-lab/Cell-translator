"""Run a healthy-differentiation or leukemia-transformation Cell Translator workflow.

The application imports the state-specific HurdleAE and acute-leukemia
CycleGAN training variants from the canonical ``cell_translator`` package;
model classes and shared training functions are not redefined here.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from cell_translator.training.independent_hurdle_ae_training import (
    IndependentHurdleAETrainingConfig,
    calibrate_global_p_threshold,
    decode_independent_latent,
    run_independent_hurdle_ae,
)
from config_utils import load_config


def _dataclass_from_section(cls, section: dict, aliases: dict[str, str] | None = None):
    aliases = aliases or {}
    valid = {f.name for f in fields(cls)}
    values = {}
    for key, value in section.items():
        name = aliases.get(key, key)
        if name in valid:
            if name == "mmd_sigmas":
                value = tuple(float(x) for x in value)
            values[name] = value
    return cls(**values)


def _hurdle_config(cfg: dict):
    return _dataclass_from_section(
        IndependentHurdleAETrainingConfig,
        cfg["hurdle_ae"],
        aliases={"learning_rate": "lr", "threshold": "threshold"},
    )


def _cyclegan_config(cfg: dict):
    # Lazy import keeps Stage 1 usable in a PyTorch-only environment.
    from cell_translator.training.acute_leukemia_cyclegan_training import AcuteLeukemiaCycleGANConfig
    return _dataclass_from_section(AcuteLeukemiaCycleGANConfig, cfg["cyclegan"])


def _validate_input(path: str, label: str, expected_cells: int, expected_genes: int):
    x = np.load(path).astype(np.float32)
    if x.shape != (expected_cells, expected_genes):
        raise ValueError(
            f"{label}: expected {(expected_cells, expected_genes)}, got {x.shape}. "
            "Update expected_input only if your public data release intentionally differs."
        )
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError(f"{label}: expected finite non-negative log1p values.")
    return x


def _library_match_log1p(fake_log1p: np.ndarray, real_log1p: np.ndarray, eps: float = 1e-8):
    fake_lin = np.expm1(fake_log1p)
    real_lin = np.expm1(real_log1p)
    lib_fake = fake_lin.sum(axis=1)
    lib_real = real_lin.sum(axis=1)
    target = float(np.median(lib_real))
    scale = target / np.maximum(lib_fake, eps)
    matched = np.log1p(fake_lin * scale[:, None]).astype(np.float32)
    return matched, target, float(np.median(lib_fake)), float(np.median(np.expm1(matched).sum(1)))


def retrain(cfg: dict):
    from cell_translator.training.acute_leukemia_cyclegan_training import train_acute_leukemia_cyclegan

    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    expected = cfg["expected_input"]
    x_src = _validate_input(
        cfg["paths"]["source_expression"], cfg["source_label"],
        int(expected["source_cells"]), int(expected["genes"]),
    )
    x_tgt = _validate_input(
        cfg["paths"]["target_expression"], cfg["target_label"],
        int(expected["target_cells"]), int(expected["genes"]),
    )
    genes = np.load(cfg["paths"]["gene_names"], allow_pickle=False).astype(str)
    if genes.size != int(expected["genes"]):
        raise ValueError(f"Expected {expected['genes']} gene names, got {genes.size}.")

    hcfg = _hurdle_config(cfg)
    source_res = run_independent_hurdle_ae(
        cfg["paths"]["source_expression"], str(out), cfg["run_names"]["source_ae"], hcfg
    )
    target_res = run_independent_hurdle_ae(
        cfg["paths"]["target_expression"], str(out), cfg["run_names"]["target_ae"], hcfg
    )

    z_src_train = source_res["Z_all"][source_res["train_idx"]].astype(np.float32)
    z_tgt_train = target_res["Z_all"][target_res["train_idx"]].astype(np.float32)
    all_train = np.concatenate([z_src_train, z_tgt_train], axis=0)
    mean_lat = all_train.mean(axis=0, keepdims=True).astype(np.float32)
    std_lat = (all_train.std(axis=0, keepdims=True) + float(cfg["translation"]["standardization_epsilon"])).astype(np.float32)
    np.save(out / "gan_mean_lat_train.npy", mean_lat)
    np.save(out / "gan_std_lat_train.npy", std_lat)

    src_train_z = (z_src_train - mean_lat) / std_lat
    tgt_train_z = (z_tgt_train - mean_lat) / std_lat
    cgan = train_acute_leukemia_cyclegan(
        src_train_z, tgt_train_z, cfg["source_state"], cfg["target_state"], _cyclegan_config(cfg)
    )
    cgan["G_fwd"].save(out / "G_forward.keras")
    cgan["G_bwd"].save(out / "G_backward.keras")
    cgan["D_tgt"].save(out / "D_target.keras")
    cgan["D_src"].save(out / "D_source.keras")
    np.savez_compressed(
        out / "cyclegan_history.npz",
        **{k: np.asarray(v, dtype=np.float32) for k, v in cgan["histories"].items()},
    )

    src_val_raw = source_res["Z_all"][source_res["val_idx"]].astype(np.float32)
    tgt_val_raw = target_res["Z_all"][target_res["val_idx"]].astype(np.float32)
    src_val_z = (src_val_raw - mean_lat) / std_lat
    tgt_val_z = (tgt_val_raw - mean_lat) / std_lat
    fake_z = cgan["G_fwd"](src_val_z, training=False).numpy().astype(np.float32)

    # Historical notebook added an additional 1e-8 when reversing the z-score.
    std_safe = std_lat + float(cfg["translation"]["unstandardization_epsilon"])
    fake_raw = fake_z * std_safe + mean_lat
    x_expected, p_fake, mag_fake = decode_independent_latent(
        target_res["model"], fake_raw, target_res["device"]
    )

    target_det = float((x_tgt > float(cfg["translation"]["hard_decode_detection_threshold"])).mean())
    p_thr = calibrate_global_p_threshold(p_fake, target_det)
    x_hard = ((p_fake > p_thr).astype(np.float32) * mag_fake).astype(np.float32)
    x_library, target_lib, before_lib, after_lib = _library_match_log1p(x_hard, x_tgt)

    translated_path = out / "translated_outputs.npz"
    np.savez_compressed(
        translated_path,
        source_z=src_val_z.astype(np.float32),
        target_z=tgt_val_z.astype(np.float32),
        fake_z=fake_z,
        fake_z_raw=fake_raw.astype(np.float32),
        X_expected=x_expected.astype(np.float32),
        X_hard=x_hard,
        X_library_matched=x_library,
        P_fake=p_fake.astype(np.float32),
        MAG_fake=mag_fake.astype(np.float32),
        p_threshold=np.asarray(p_thr, dtype=np.float32),
        target_detection_fraction=np.asarray(target_det, dtype=np.float32),
        mean_lat_train=mean_lat,
        std_lat_train=std_lat,
        source_val_idx=source_res["val_idx"].astype(np.int64),
        target_val_idx=target_res["val_idx"].astype(np.int64),
        gene_names=genes,
    )
    metadata = {
        "workflow": cfg["workflow"],
        "target_detection_fraction": target_det,
        "p_threshold": p_thr,
        "hard_detection_fraction": float((x_hard > 0).mean()),
        "target_median_library_size": target_lib,
        "translated_median_library_before_matching": before_lib,
        "translated_median_library_after_matching": after_lib,
        "balanced_cyclegan_training_cells": cgan["n_balanced_training_cells"],
    }
    (out / "translation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved translated outputs: {translated_path}")
    return translated_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["retrain", "evaluate"], default="retrain")
    ap.add_argument("--translated", help="Existing translated_outputs.npz for --mode evaluate")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.mode == "retrain":
        translated = retrain(cfg)
    else:
        if not args.translated:
            raise SystemExit("--mode evaluate requires --translated.")
        translated = Path(args.translated)

    from evaluation import evaluate
    metrics, arrays = evaluate(cfg, str(translated))
    out = Path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(out / "log2fc_arrays.npz", **arrays)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
