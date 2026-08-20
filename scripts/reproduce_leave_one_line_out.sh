#!/usr/bin/env bash
set -euo pipefail
# Retraining path from annotated morphology inputs:
python -m analyses.leave_one_line_out.train --config configs/validation/leave_one_line_out.yaml --transition all

# Final corrected post-hoc metrics from archived five-line fold outputs:
python -m analyses.leave_one_line_out.evaluate_archived_folds --config configs/validation/leave_one_line_out.yaml
python -m analyses.leave_one_line_out.plot \
  --results results/leave_one_line_out/archived_evaluation/lodo_five_line_corrected_metrics.csv \
  --outdir results/leave_one_line_out/figures
