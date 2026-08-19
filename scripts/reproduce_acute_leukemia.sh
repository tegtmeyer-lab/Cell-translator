#!/usr/bin/env bash
set -euo pipefail

python applications/acute_leukemia/run.py \
  --config configs/acute_leukemia/healthy_cd34_to_bmmc.yaml \
  --mode retrain

python applications/acute_leukemia/validate_results.py \
  --metrics results/acute_leukemia/healthy_cd34_to_bmmc/evaluation_metrics.json \
  --expected configs/acute_leukemia/expected_results.yaml \
  --workflow healthy_cd34_to_bmmc

python applications/acute_leukemia/run.py \
  --config configs/acute_leukemia/disease_cd34_to_mpal.yaml \
  --mode retrain

python applications/acute_leukemia/validate_results.py \
  --metrics results/acute_leukemia/disease_cd34_to_mpal/evaluation_metrics.json \
  --expected configs/acute_leukemia/expected_results.yaml \
  --workflow disease_cd34_to_mpal
