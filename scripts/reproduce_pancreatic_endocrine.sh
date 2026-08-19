#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/pancreatic_endocrine/gse132188_ngn3_to_endocrine.yaml}"

python applications/pancreatic_endocrine/run.py   --config "$CONFIG"   --all   --no-show-plots
