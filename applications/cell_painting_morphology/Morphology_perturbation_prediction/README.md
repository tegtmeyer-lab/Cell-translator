# Cell Translator — morphology perturbation prediction

This repository contains the morphology-perturbation application of **Cell Translator**, comparing high-dimensional morphological responses in induced pluripotent stem cells (iPSCs) and astrocytes under DMSO and three pharmacological perturbations: **Latrunculin B, GW3965, and Clozapine**.

The workflow includes (1) condition-specific morphology autoencoders, (2) latent-space developmental translation from iPSCs to astrocytes (four conditions including DMSO), and (3) within-cell-type DMSO-to-drug translation (three drugs × two cell types). Evaluation includes morphology effect prediction, drug specificity, top-feature overlap, transition-vector geometry, morphology-family enrichment, and integrated-gradients attribution.

> **Status:** Research code accompanying an analysis; not a turnkey library or independently reproduced benchmark. The notebooks have been cleaned of outputs and machine-specific paths. Code syntax and path substitution were checked, but the complete training/evaluation pipeline has **not** been executed against the private input data in this package. Do not claim reproduced results without running the workflow.

## Repository structure

```text
notebooks/
  01_train_morphology_translators.ipynb
  02_evaluate_drug_effects.ipynb
  03_feature_family_enrichment_and_IG.ipynb
  04_training_and_reconstruction_plots.ipynb
scripts/
  morphology_transition_vectors.py
  plot_family_enrichment.py
  plot_cross_drug_specificity.py
  plot_topk_feature_recovery.py
requirements.txt
environment.example
.gitignore
```

## Input data

The training notebook expects eight **private, preprocessed single-cell morphology tables** with matching feature names, one row per cell, and numeric CellProfiler-derived morphology feature columns. Metadata/positional columns are filtered by the notebook; the raw images and underlying processed measurement files are **not included**.

Put these CSV files into `data/` or point `CELL_TRANSLATOR_DATA_DIR` at their directory:

```text
iPSC_DMSO.csv
iPSC_Latrunculin_B.csv
iPSC_GW3965.csv
iPSC_Clozapine.csv
Astrocytes_DMSO.csv
Astrocytes_Latrunculin_B.csv
Astrocytes_GW3965.csv
Astrocytes_Clozapine.csv
```

**Important:** Ensure you have permission to publish any input data before adding it to a public repository. This project deliberately excludes those files and all trained weights, cell profiles, and generated figures.

## Installation

Create a clean Python environment appropriate for your CUDA/CPU, PyTorch, and TensorFlow versions. The requirements are an **unlocked dependency list** rather than a tested environment lockfile.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your private paths before starting Jupyter:

```bash
export CELL_TRANSLATOR_DATA_DIR="/path/to/my/morphology-input-csvs"
export CELL_TRANSLATOR_RUN_DIR="/path/to/my/morphology-output-directory"
jupyter lab
```

If neither variable is set, the default input is the repository's `data/` and the default output is `outputs/rerun_70_15_15_qc_v3/`. **Start Jupyter from the repository root** for the relative defaults to resolve correctly. The notebooks read these environment variables directly; no path editing is necessary.

## Suggested execution order

1. Run `notebooks/01_train_morphology_translators.ipynb` **top-to-bottom** to audit inputs, create the 70/15/15 partitions, train eight autoencoders, and run the ten specified translations. The folder named `rerun_70_15_15_qc_v3` is merely the historical run identifier; it does not imply your independent execution is validated.
2. Run `notebooks/02_evaluate_drug_effects.ipynb` to evaluate perturbation recovery, feature-ranking overlap, and compound specificity.
3. Run `notebooks/03_feature_family_enrichment_and_IG.ipynb` to compute feature-family over/underrepresentation and source-feature IG enrichment. This analysis needs the trained model artifacts created in step 1.
4. Optionally run `notebooks/04_training_and_reconstruction_plots.ipynb` for training curves and autoencoder diagnostics.
5. Generate figures using the scripts below:

```bash
python scripts/morphology_transition_vectors.py
python scripts/plot_topk_feature_recovery.py
python scripts/plot_cross_drug_specificity.py
python scripts/plot_family_enrichment.py
```

Each figure script reads existing locally generated model or evaluation files and **does not train models**. Scripts may need different prerequisites (e.g., the cross-drug heatmap requires the two saved correlation matrices from the evaluation notebook).

## Methods notes

- The corrected training notebook uses **70% training / 15% validation / 15% test** within each condition. Training data determine feature filtering, imputation and standardization parameters. Autoencoder validation is for early stopping and test data for final evaluation.
- `GROUP_COLUMN = None` is the default, corresponding to **cell-level splitting**, not donor-, plate-, cell-line-, or experiment-independent generalization. Set a relevant grouping column **before training** if your data permit group-held-out evaluation, and verify assignment across conditions.
- The TensorFlow latent CycleGAN training uses the notebook's existing fixed-epoch training design; do not describe it as validation-selected checkpointing unless separately implemented.
- Drug-effect evaluation should compare **aligned feature names**, with treatment contrasts relative to DMSO in the same cell type. PCA in `morphology_transition_vectors.py` is only for visualization; the reported transition-vector metrics are computed in a shared standardized morphology space.
- Integrated gradients describe **model attribution**, not biological causality or the signed direction of the drug effect. Hypergeometric family tests operate on correlated feature sets and are not independent biological-replicate tests.
- Drug conditions appear during training; results are **not evidence of generalization to unseen drugs**.

## Reproducibility and sharing

Source data, scaler statistics, model checkpoints, predictions and evaluation CSVs are **not shipped**. The code expects the original preprocessed morphology-feature tables, and additional provenance of those tables (CellProfiler pipeline, normalization, and upstream quality-control rules) is needed to reproduce the work from raw images. Publish the software version details and data-availability statement associated with the manuscript if available.

## Citation and authorship

If you use or adapt this code, please cite the associated **Cell Translator** manuscript when it becomes available. Add the final paper title, author list, DOI/preprint link and license **after confirming them with the project maintainers**. No license is assigned here; obtain agreement from relevant rights holders before making the repository public.
