"""Pancreatic endocrine differentiation application runner.

This module defines the NGN3-high-EP to Endocrine workflow and imports the
shared HurdleAE, CycleGAN, and inference implementation from
``cell_translator``.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from cell_translator.training import HurdleAETrainingConfig, run_shared_hurdle_ae
from config_utils import load_config
from evaluation import evaluate_translation, print_metrics, save_metrics


def prepared_paths(cfg: dict) -> dict[str, str]:
    root = Path(cfg["paths"]["prepared_dir"])
    return {
        cfg["source_state"]: str(root / f"{cfg['source_state']}_X_hvg.npy"),
        cfg["target_state"]: str(root / f"{cfg['target_state']}_X_hvg.npy"),
    }


def artifact_paths(cfg: dict) -> tuple[str, str, str]:
    out = Path(cfg["paths"]["output_dir"])
    run_name = cfg["run_name"]
    slug = f"{cfg['source_state'].lower()}_to_{cfg['target_state'].lower()}".replace(" ", "_")
    return (
        str(out / f"{run_name}_outputs.npz"),
        str(out / slug / "translated_source_outputs.npz"),
        str(out / "evaluation_metrics.json"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train-ae", action="store_true")
    parser.add_argument("--train-gan", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-show-plots", action="store_true")
    parser.add_argument(
        "--translated",
        default=None,
        help="Optional archived translated .npz to evaluate instead of a newly trained output.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.all:
        args.prepare = args.train_ae = args.train_gan = args.evaluate = True

    if args.prepare:
        from prepare_data import prepare_dataset
        prepare_dataset(cfg)

    x_paths = prepared_paths(cfg)
    state_names = [cfg["source_state"], cfg["target_state"]]
    Path(cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    if args.train_ae:
        run_shared_hurdle_ae(
            x_paths=x_paths,
            state_names=state_names,
            save_dir=cfg["paths"]["output_dir"],
            run_name=cfg["run_name"],
            latent_dim=int(cfg["model"]["latent_dim"]),
            training_config=HurdleAETrainingConfig.from_dict(cfg["hurdle_ae"]),
        )

    ae_npz, translated_default, metrics_path = artifact_paths(cfg)
    if args.train_gan:
        # Import TensorFlow-dependent inference only when Stage 2 is requested,
        # so preparation and HurdleAE-only workflows remain usable without TensorFlow.
        from cell_translator.inference.translate import run_translation
        run_translation(
            ae_outputs_npz=ae_npz,
            x_paths=x_paths,
            state_names=state_names,
            source_state=cfg["source_state"],
            target_state=cfg["target_state"],
            save_dir=cfg["paths"]["output_dir"],
            plot=not args.no_show_plots,
            cyclegan_config=cfg["cyclegan"],
            translation_split=cfg["translation"]["split"],
        )

    if args.evaluate:
        metrics = evaluate_translation(
            target_matrix=x_paths[cfg["target_state"]],
            ae_npz=ae_npz,
            translated_npz=args.translated or translated_default,
            target_state=cfg["target_state"],
            threshold=float(cfg["evaluation"]["threshold"]),
            seed=int(cfg["evaluation"]["seed"]),
        )
        print_metrics(metrics)
        save_metrics(metrics, metrics_path)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
