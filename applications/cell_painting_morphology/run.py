"""Cell Painting morphology translation application runner.

The application uses a dense morphology autoencoder and a morphology-specific
latent CycleGAN; it does not use the transcriptomic HurdleAE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from cell_translator.training.morphology_ae_training import (
    MorphologyAETrainingConfig,
    run_morphology_ae_state,
    load_state_artifacts,
    inverse_scale,
)
from cell_translator.training.morphology_cyclegan_training import (
    MorphologyCycleGANConfig,
    train_morphology_cyclegan,
)
from config_utils import load_config
from prepare_data import prepare_data
from evaluation import evaluate_translation, save_metrics


def _transition_by_label(cfg, label):
    transitions = [t for t in cfg["transitions"] if t.get("enabled", True)]
    if label is None:
        return transitions
    matches = [t for t in transitions if t["label"] == label]
    if not matches:
        raise ValueError(f"Unknown or disabled transition: {label}")
    return matches


def _train_all_aes(cfg):
    prepared = Path(cfg["paths"]["prepared_dir"])
    output = Path(cfg["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(prepared / "common_feature_columns.csv")["feature"].astype(str).tolist()
    ae_cfg = MorphologyAETrainingConfig.from_dict(cfg["morphology_ae"])
    summary = []
    for state_name in cfg["states"]:
        X = np.load(prepared / f"{state_name}_X.npy")
        result = run_morphology_ae_state(X, features, state_name, str(output), ae_cfg)
        row = {
            "state": state_name,
            "n_cells": result["n_cells"],
            "best_epoch": result["best_epoch"],
            "best_val": result["best_val"],
            **result["validation"],
        }
        summary.append(row)
    pd.DataFrame(summary).to_csv(output / "ae_validation_summary.csv", index=False)


def _evaluate_transition(cfg, transition):
    output = Path(cfg["paths"]["output_dir"])
    source, target = transition["source"], transition["target"]
    run_dir = output / f"{source}_to_{target}"
    X_source = np.load(run_dir / "source_validation_features.npy")
    X_target = np.load(run_dir / "target_validation_features.npy")
    X_translated = np.load(run_dir / "translated_features.npy")

    # Some historical validation blocks trained the target-vs-source k-NN
    # classifier on the measured AE training split. Reconstruct those
    # matrices on the original Cell Painting scale from the saved train-only
    # scaler statistics rather than re-reading raw input data.
    source_artifacts = load_state_artifacts(source, str(output))
    target_artifacts = load_state_artifacts(target, str(output))
    X_source_train = inverse_scale(
        source_artifacts["X_train_scaled"], source_artifacts["scaler_stats"]
    )
    X_target_train = inverse_scale(
        target_artifacts["X_train_scaled"], target_artifacts["scaler_stats"]
    )

    eval_cfg = dict(cfg.get("evaluation", {}))
    eval_cfg.update(transition.get("evaluation", {}))
    metrics, ks = evaluate_translation(
        X_source,
        X_target,
        X_translated,
        seed=int(eval_cfg.pop("seed", 42)),
        X_source_train=X_source_train,
        X_target_train=X_target_train,
        **eval_cfg,
    )
    metrics["transition"] = transition["label"]
    save_metrics(metrics, run_dir / "evaluation_metrics.json")
    np.save(run_dir / "per_feature_ks.npy", ks)
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--transition", default=None, help="Optional exact transition label from the YAML file.")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train-ae", action="store_true")
    parser.add_argument("--train-gan", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.all:
        args.prepare = args.train_ae = args.train_gan = args.evaluate = True

    if args.prepare:
        prepare_data(cfg)
    if args.train_ae:
        _train_all_aes(cfg)

    transitions = _transition_by_label(cfg, args.transition)
    gan_cfg = MorphologyCycleGANConfig.from_dict(cfg["cyclegan"])
    if args.train_gan:
        for transition in transitions:
            print(f"Training {transition['label']}")
            train_morphology_cyclegan(
                source_state=transition["source"],
                target_state=transition["target"],
                artifact_dir=cfg["paths"]["output_dir"],
                output_dir=cfg["paths"]["output_dir"],
                config=gan_cfg,
            )
    if args.evaluate:
        rows = [_evaluate_transition(cfg, transition) for transition in transitions]
        pd.DataFrame(rows).to_csv(
            Path(cfg["paths"]["output_dir"]) / "translation_validation_summary.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
