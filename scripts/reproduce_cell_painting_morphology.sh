#!/usr/bin/env bash
set -euo pipefail

CONTROL_CONFIG=${1:-configs/cell_painting_morphology/control_line24.yaml}
DELETION_CONFIG=${2:-configs/cell_painting_morphology/deletion_line45.yaml}

python applications/cell_painting_morphology/run.py --config "$CONTROL_CONFIG" --all
python applications/cell_painting_morphology/run.py --config "$DELETION_CONFIG" --all

python applications/cell_painting_morphology/effect_size_comparison.py \
  --control-config "$CONTROL_CONFIG" \
  --deletion-config "$DELETION_CONFIG"
