#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/benchmarks/perturbation_tools.yaml}
CELLOT_PYTHON=${CELLOT_PYTHON:-python}
SCGEN_PYTHON=${SCGEN_PYTHON:-python}
SCGPT_PYTHON=${SCGPT_PYTHON:-python}

# Use separate interpreters/environments when required, e.g.
# CELLOT_PYTHON=/path/to/cellot/bin/python \
# SCGEN_PYTHON=/path/to/scgen/bin/python \
# SCGPT_PYTHON=/path/to/scgpt/bin/python \
# bash scripts/reproduce_perturbation_tool_benchmarks.sh

"$CELLOT_PYTHON" -m analyses.perturbation_tool_benchmarks.cellot_prepare --config "$CONFIG"
"$CELLOT_PYTHON" -m analyses.perturbation_tool_benchmarks.cellot_train --config "$CONFIG"
"$CELLOT_PYTHON" -m analyses.perturbation_tool_benchmarks.cellot_export --config "$CONFIG"
"$SCGEN_PYTHON" -m analyses.perturbation_tool_benchmarks.scgen_runner --config "$CONFIG"
"$SCGPT_PYTHON" -m analyses.perturbation_tool_benchmarks.scgpt_postprocess --config "$CONFIG"

echo "Benchmark prediction exports complete."
echo "Update configs/genetic_perturbation/comparison.yaml with the generated output paths, then run compare_models.py."
