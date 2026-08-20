#!/usr/bin/env bash
set -euo pipefail
python -m analyses.baseline_comparison.run --config configs/validation/baseline_comparison.yaml
python -m analyses.baseline_comparison.plot \
  --results results/baseline_comparison/baseline_comparison_all_metrics.csv \
  --outdir results/baseline_comparison/figures
