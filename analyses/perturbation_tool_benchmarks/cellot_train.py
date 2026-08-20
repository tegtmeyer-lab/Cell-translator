"""Invoke an external CellOT checkout using the benchmark task configuration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .common import PERTURBATIONS, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--gene", choices=PERTURBATIONS)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    ccfg = cfg["cellot"]

    repo = Path(ccfg["upstream_repo"])
    train_py = repo / "scripts" / "train.py"
    evaluate_py = repo / "scripts" / "evaluate.py"
    model_cfg = repo / ccfg.get("model_config_relative", "configs/models/cellot.yaml")
    task_cfg = Path(ccfg["output_root"]) / "task_pca50.yaml"
    for required in (train_py, evaluate_py, model_cfg, task_cfg):
        if not required.exists():
            raise FileNotFoundError(required)

    genes = [args.gene] if args.gene else list(cfg["perturbations"])
    for gene in genes:
        outdir = Path(ccfg["output_root"]) / gene / "model-cellot-pca50"
        outdir.mkdir(parents=True, exist_ok=True)
        train_cmd = [
            sys.executable, str(train_py),
            "--outdir", str(outdir),
            "--config", str(task_cfg),
            "--config", str(model_cfg),
            "--config.data.target", gene,
            "--config.training.n_iters", str(int(ccfg["n_iters"])),
        ]
        eval_cmd = [
            sys.executable, str(evaluate_py),
            "--outdir", str(outdir),
            "--setting", str(ccfg["evaluation_setting"]),
            "--where", str(ccfg["evaluation_space"]),
        ]
        print("Running:", " ".join(train_cmd))
        subprocess.run(train_cmd, check=True)
        print("Running:", " ".join(eval_cmd))
        subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
