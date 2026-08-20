# Perturbation baseline comparison

This analysis compares archived Cell Translator perturbation predictions with four simple baselines: Identity, MeanShift, NearestNeighbor, and LinearLatentShift. It preserves the evaluation definitions from the source analysis.

The comparison is distribution-level and unpaired. Source and measured-target matrices are independently split 70:30 using random state 42; the simple baselines learn only from the 70% training subsets. Archived Cell Translator predictions are subsampled to the same number of evaluation source cells when needed.

Key settings are stored in `configs/validation/baseline_comparison.yaml`: 50 PCA components, top 500 effect genes, up to 3,000 genes for KS/Wasserstein metrics, and 1,000 cells per class for the discriminator AUC. Identity has no perturbation effect, so effect-correlation and top-effect sign-agreement values are displayed as NA in the final comparison table/figures.

Run:

```bash
python analyses/baseline_comparison/run.py \
  --config configs/validation/baseline_comparison.yaml

python analyses/baseline_comparison/plot.py \
  --results results/baseline_comparison/baseline_comparison_all_metrics.csv \
  --outdir results/baseline_comparison/figures
```


