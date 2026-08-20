"""Genetic perturbation application runner.

This module defines the application-specific control-to-perturbation workflow
and imports the shared Cell Translator model and training components from
``cell_translator``.

It supports two execution modes:

``archived``
    Recomputes predictions from previously trained model artifacts and is the
    preferred mode for reproducing the reported numerical results.

``retrain``
    Retrains Stage 1 and Stage 2 from the processed expression matrices using
    the parameters specified in the YAML configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cell_translator.inference.perturbation import (
    decode_target_latents,
    load_perturbation_ae_checkpoint,
    load_perturbation_latents,
    predict_with_archived_generator,
)
from cell_translator.training.cyclegan_training import CycleGANTrainingConfig, train_latent_cyclegan
from cell_translator.training.perturbation_hurdle_ae_training import run_perturbation_hurdle_ae
from config_utils import load_config
from evaluation import evaluate_heldout_translation


def _validate_inputs(cfg):
    expected = cfg.get("expected_input", {})
    X_ctrl = np.load(cfg["data"]["control_expression"], mmap_mode="r")
    X_target = np.load(cfg["data"]["target_expression"], mmap_mode="r")
    if X_ctrl.shape[1] != X_target.shape[1]:
        raise ValueError("Control and perturbation matrices have different gene dimensions.")
    checks = [
        ("control_cells", X_ctrl.shape[0]),
        ("target_cells", X_target.shape[0]),
        ("genes", X_ctrl.shape[1]),
    ]
    for key, actual in checks:
        if key in expected and int(expected[key]) != int(actual):
            raise ValueError(f"Input check failed for {key}: expected {expected[key]}, observed {actual}")
    return np.asarray(X_ctrl), np.asarray(X_target)


def _save_predictions(out_dir, gene, latent_pred, decoded, source_indices, target_indices):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "translated_validation_outputs.npz"
    np.savez_compressed(
        path,
        gene=np.asarray(gene),
        source_validation_indices=np.asarray(source_indices, dtype=np.int64),
        target_validation_indices=np.asarray(target_indices, dtype=np.int64),
        translated_target_latent_standardized=np.asarray(latent_pred, dtype=np.float32),
        translated_target_latent=np.asarray(decoded["Z_target"], dtype=np.float32),
        X_pred_soft=np.asarray(decoded["X_soft"], dtype=np.float32),
        X_pred_hard=np.asarray(decoded["X_hard"], dtype=np.float32),
    )
    return path


def run_archived(cfg):
    gene = cfg["gene"]
    out_dir = Path(cfg["outputs"]["directory"])
    X_ctrl, X_target = _validate_inputs(cfg)
    lat = load_perturbation_latents(cfg["archived_artifacts"]["ae_outputs"], gene)
    model, _ = load_perturbation_ae_checkpoint(
        cfg["archived_artifacts"]["ae_checkpoint"], gene, input_dim=X_ctrl.shape[1]
    )
    _, Z_pred = predict_with_archived_generator(
        cfg["archived_artifacts"]["forward_generator"], lat["val_src"]
    )
    decoded = decode_target_latents(
        model,
        Z_pred,
        lat["mean_target"],
        lat["std_target"],
        lat["tau_target"],
    )
    pred_path = _save_predictions(
        out_dir, gene, Z_pred, decoded, lat["ctrl_val_idx"], lat["target_val_idx"]
    )
    summary = evaluate_heldout_translation(
        X_ctrl[lat["ctrl_val_idx"]],
        X_target[lat["target_val_idx"]],
        decoded["X_hard"],
        np.load(cfg["data"]["gene_names"], allow_pickle=False),
        threshold=float(cfg["evaluation"]["threshold"]),
        output_dir=out_dir / "evaluation",
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved predictions: {pred_path}")


def run_retrain(cfg):
    gene = cfg["gene"]
    out_dir = Path(cfg["outputs"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    X_ctrl, X_target = _validate_inputs(cfg)

    ae_dir = out_dir / "hurdle_ae"
    ae = run_perturbation_hurdle_ae(
        cfg["data"]["control_expression"],
        cfg["data"]["target_expression"],
        gene,
        str(ae_dir),
        run_name=f"perturbation_hurdle_ae_{gene}",
        config=cfg["hurdle_ae"],
    )
    lat = load_perturbation_latents(ae["outputs_path"], gene)
    gan_cfg = CycleGANTrainingConfig.from_dict(cfg["cyclegan"])
    gan = train_latent_cyclegan(
        lat["train_src"], lat["train_tgt"], "control", gene.lower(), config=gan_cfg
    )
    gan_dir = out_dir / "cyclegan"
    gan_dir.mkdir(parents=True, exist_ok=True)
    gan["G_fwd"].save(gan_dir / f"G_control_to_{gene.lower()}.keras")
    gan["G_bwd"].save(gan_dir / f"G_{gene.lower()}_to_control.keras")
    gan["D_tgt"].save(gan_dir / f"D_{gene.lower()}.keras")
    gan["D_src"].save(gan_dir / "D_control.keras")
    np.save(gan_dir / "training_histories.npy", gan["histories"], allow_pickle=True)
    (gan_dir / "training_config.json").write_text(
        json.dumps(gan["training_config"], indent=2)
    )

    Z_pred_train = gan["G_fwd"].predict(lat["train_src"], batch_size=256, verbose=0).astype(np.float32)
    Z_pred = gan["G_fwd"].predict(lat["val_src"], batch_size=256, verbose=0).astype(np.float32)
    # Preserve all arrays needed by the original downstream result suites, while
    # using generic names that work for every perturbation.
    np.savez_compressed(
        gan_dir / "cyclegan_outputs.npz",
        Z_pred_target_train_std=Z_pred_train,
        Z_pred_target_val_std=Z_pred,
        Z_source_train_std=lat["train_src"],
        Z_source_val_std=lat["val_src"],
        Z_target_train_std=lat["train_tgt"],
        Z_target_val_std=lat["val_tgt"],
        ctrl_tr_idx=lat["ctrl_tr_idx"],
        ctrl_te_idx=lat["ctrl_val_idx"],
        target_tr_idx=lat["target_tr_idx"],
        target_te_idx=lat["target_val_idx"],
        **{f"{gene.lower()}_tr_idx": lat["target_tr_idx"], f"{gene.lower()}_te_idx": lat["target_val_idx"]},
    )
    decoded = decode_target_latents(
        ae["model"], Z_pred, lat["mean_target"], lat["std_target"], lat["tau_target"],
        device=ae["device"],
    )
    _save_predictions(out_dir, gene, Z_pred, decoded, lat["ctrl_val_idx"], lat["target_val_idx"])
    summary = evaluate_heldout_translation(
        X_ctrl[lat["ctrl_val_idx"]],
        X_target[lat["target_val_idx"]],
        decoded["X_hard"],
        np.load(cfg["data"]["gene_names"], allow_pickle=False),
        threshold=float(cfg["evaluation"]["threshold"]),
        output_dir=out_dir / "evaluation",
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene", required=True, choices=["APOE", "MSI2", "NR1H3", "LRRC4C", "ANKRD12"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["archived", "retrain"], default="archived")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("gene") != args.gene:
        raise ValueError(f"--gene {args.gene} does not match config gene {cfg.get('gene')}")
    if args.mode == "archived":
        run_archived(cfg)
    else:
        run_retrain(cfg)


if __name__ == "__main__":
    main()
