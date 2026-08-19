"""Neuronal differentiation application runner.

This module defines the neuronal differentiation workflow and imports the
shared HurdleAE, CycleGAN, loss functions, training routines, and inference
utilities from the ``cell_translator`` package.
"""

import argparse
import os
from pathlib import Path

import yaml

from cell_translator.training import run_shared_hurdle_ae
from cell_translator.inference.translate import run_translation


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to application YAML configuration")
    parser.add_argument(
        "--translation",
        choices=["ipsc_to_npc", "npc_to_neuron", "npc_to_astro", "all"],
        default="all",
    )
    parser.add_argument("--train-ae", action="store_true", help="Train required HurdleAE run(s)")
    parser.add_argument("--no-show-plots", action="store_true", help="Save plots without opening a GUI window")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data_paths = {k: str(v) for k, v in cfg["data"].items()}
    save_dir = str(cfg["output_dir"])
    os.makedirs(save_dir, exist_ok=True)

    requested = list(cfg["translations"]) if args.translation == "all" else [args.translation]
    required_ae_runs = {cfg["translations"][name]["ae_run"] for name in requested}

    if args.train_ae:
        for ae_key in required_ae_runs:
            ae_cfg = cfg["ae_runs"][ae_key]
            states = list(ae_cfg["states"])
            x_paths = {state: data_paths[state] for state in states}
            run_shared_hurdle_ae(
                x_paths=x_paths,
                state_names=states,
                save_dir=save_dir,
                run_name=str(ae_cfg["run_name"]),
                latent_dim=int(ae_cfg["latent_dim"]),
            )

    for name in requested:
        tcfg = cfg["translations"][name]
        ae_cfg = cfg["ae_runs"][tcfg["ae_run"]]
        states = list(ae_cfg["states"])
        x_paths = {state: data_paths[state] for state in states}
        ae_npz = os.path.join(save_dir, str(ae_cfg["run_name"]) + "_outputs.npz")
        run_translation(
            ae_outputs_npz=ae_npz,
            x_paths=x_paths,
            state_names=states,
            source_state=str(tcfg["source_state"]),
            target_state=str(tcfg["target_state"]),
            save_dir=save_dir,
            plot=not args.no_show_plots,
        )


if __name__ == "__main__":
    main()
