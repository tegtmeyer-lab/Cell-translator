# Genetic perturbation application

This application reproduces the Cell Translator control-to-perturbation analyses
for **APOE, MSI2, NR1H3, LRRC4C, and ANKRD12**.

The biological application uses the common Cell Translator latent CycleGAN but a
perturbation-specific Stage-1 HurdleAE architecture: control and perturbation
cells have **separate encoders** coupled to one **shared Hurdle decoder**. The
architecture, loss functions, split strategy, training-loop behavior, and
hyperparameters are centralized under `cell_translator/` and are not redefined
in the application scripts.

## Input data

The preparation workflow expects one AnnData (`.h5ad`) object containing the
control and five perturbation conditions in a common gene space. The source
analysis used an object that was already normalized and log1p-transformed.
Therefore this script performs **no additional normalization and no HVG
filtering**; all genes are retained.

Expected source-analysis dimensions are recorded in
`configs/genetic_perturbation/expected_results.yaml`:

- control: 10,529 cells
- APOE: 681 cells
- MSI2: 684 cells
- NR1H3: 681 cells
- LRRC4C: 8,828 cells
- ANKRD12: 568 cells
- genes: 18,614

All paths in the public YAML files are placeholders and must be replaced with
local data/artifact paths.

## 1. Prepare the matrices

```bash
python applications/genetic_perturbation/prepare_data.py \
  --config configs/genetic_perturbation/prepare_data.yaml
```

The cleaned workflow stores the shared control matrix only once rather than
copying it into five pairwise directories. This changes storage organization
only; the expression values supplied to the models are unchanged. 

## 2. Recompute predictions from the archived models

For numerical reproduction of the reported predictions, use the archived
HurdleAE and Keras generator artifacts produced by the original analysis:

```bash
python applications/genetic_perturbation/run.py \
  --gene APOE \
  --config configs/genetic_perturbation/apoe.yaml \
  --mode archived
```

Replace `APOE` and its YAML with `MSI2`, `NR1H3`, `LRRC4C`, or `ANKRD12` for the
other applications.

## 3. Retrain from the processed matrices

```bash
python applications/genetic_perturbation/run.py \
  --gene APOE \
  --config configs/genetic_perturbation/apoe.yaml \
  --mode retrain
```

The perturbation CycleGAN configuration intentionally preserves the settings
used in this application, including `BANK_SAMPLE=512`, RBF-MMD bandwidths
`[4, 8, 16, 32]`, source-based steps per epoch, and the original unseeded
TensorFlow behavior (`tf_seed: null`). These differ from some other Cell
Translator applications and are specified explicitly in `common.yaml`.

## 4. Evaluate the translated held-out cells

`run.py` automatically writes:

```text
translated_validation_outputs.npz
evaluation/heldout_metrics.json
evaluation/per_gene_ks.csv
evaluation/per_gene_wasserstein.csv
evaluation/effect_summaries.npz
```

The single-perturbation result suite uses held-out control and held-out target
cells.

## 5. Recreate the consolidated single-perturbation figures

The repeated per-gene result-suite cells were consolidated into one generic
script. It recreates the PCA/t-SNE/UMAP latent-space comparison, real-versus-
predicted mean/detection correlations, per-gene KS and Wasserstein plots, the
historical LFC agreement plot, and marker-gene violins using the saved
held-out predictions.

```bash
python applications/genetic_perturbation/results_suite.py \
  --gene APOE \
  --config configs/genetic_perturbation/apoe.yaml
```

The marker lists used by each original single-perturbation result suite are
recorded in the gene-specific YAML files.

## 6. Evaluate Stage-1 reconstruction

```bash
python applications/genetic_perturbation/evaluate_ae.py \
  --gene APOE \
  --config configs/genetic_perturbation/apoe.yaml
```

This reports gene-mean and detection-rate reconstruction correlations and saves
per-gene reconstruction metrics for the soft and hard HurdleAE decodes.

## 7. Dedicated linear-space log2FC analysis

The final repeated log2FC analysis converts log1p values back to linear space,
computes per-gene means, and then calculates the log2 ratio with pseudocount
`1e-3`.

```bash
python applications/genetic_perturbation/log2fc_analysis.py \
  --gene APOE \
  --config configs/genetic_perturbation/apoe.yaml \
  --predictions path/to/APOE/translated_validation_outputs.npz
```

## 8. Integrated-gradients interpretation

```bash
python applications/genetic_perturbation/interpret.py \
  --gene APOE \
  --config configs/genetic_perturbation/apoe.yaml \
  --predictions path/to/APOE/translated_validation_outputs.npz
```

The integrated-gradients objective and driverscore formula are
preserved. DriverScore weights and filtering thresholds are read from YAML.

## 9. Compare Cell Translator with baseline models

After replacing the placeholder paths in `comparison.yaml`:

```bash
python applications/genetic_perturbation/compare_models.py \
  --config configs/genetic_perturbation/comparison.yaml
```

This reproduces the final summary metrics based on per-gene KS statistics,
detection-rate recovery, and effect recovery among the top 5,000 measured DE
genes.

The separate final gene-mean distribution summary can be reproduced with:

```bash
python applications/genetic_perturbation/aggregate_metrics.py \
  --config configs/genetic_perturbation/comparison.yaml
```

These are distinct analyses: the latter applies KS/Wasserstein tests to the
**distribution of gene means**, whereas the model-comparison summary evaluates
**per-gene cell-level distributions**.

## 10. Validate against reference values

Reference values stored only for checking
reproduction; they are never consumed by model training.

```bash
python applications/genetic_perturbation/validate_results.py \
  --gene-config configs/genetic_perturbation/apoe.yaml
```

For the cross-model summary:

```bash
python applications/genetic_perturbation/validate_results.py \
  --comparison-csv path/to/model_comparison_summary.csv
```

## Reproducibility notes

- The Stage-1 split is performed independently within control and target using
  `test_size=0.20` and seed 42. This 20% subset is a validation/held-out subset.
- The HurdleAE training loop intentionally consumes the two training loaders
  with `zip()` and `drop_last=True`; the smaller perturbation domain therefore
  determines the number of Stage-1 updates per epoch. 

