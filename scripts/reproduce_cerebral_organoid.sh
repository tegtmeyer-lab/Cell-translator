#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/cerebral_organoid/primary_scp1756.yaml}"

python applications/cerebral_organoid/run.py \
  --config "$CONFIG" \
  --all \
  --no-show-plots
