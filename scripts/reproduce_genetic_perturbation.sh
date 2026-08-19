#!/usr/bin/env bash
set -euo pipefail

# Replace placeholder paths in configs/genetic_perturbation/*.yaml before use.
# Archived mode is the numerical reproduction path for reported predictions.

for gene in APOE MSI2 NR1H3 LRRC4C ANKRD12; do
  cfg="configs/genetic_perturbation/$(echo "$gene" | tr '[:upper:]' '[:lower:]').yaml"
  echo "=== ${gene} ==="
  python applications/genetic_perturbation/run.py --gene "$gene" --config "$cfg" --mode archived
  python applications/genetic_perturbation/evaluate_ae.py --gene "$gene" --config "$cfg"
  python applications/genetic_perturbation/log2fc_analysis.py --gene "$gene" --config "$cfg"
  python applications/genetic_perturbation/validate_results.py --gene-config "$cfg"
done

python applications/genetic_perturbation/compare_models.py \
  --config configs/genetic_perturbation/comparison.yaml
python applications/genetic_perturbation/aggregate_metrics.py \
  --config configs/genetic_perturbation/comparison.yaml

COMPARISON_OUT=$(python - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path("configs/genetic_perturbation/comparison.yaml").read_text())
print(cfg["output_dir"])
PY
)
python applications/genetic_perturbation/validate_results.py \
  --comparison-csv "$COMPARISON_OUT/model_comparison_summary.csv" \
  --aggregate-csv "$COMPARISON_OUT/celltranslator_aggregate_gene_mean_metrics.csv"
