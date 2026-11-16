# Cell Translator - morphology perturbation prediction

This repository contains the morphology-perturbation application of **Cell Translator**, comparing high-dimensional morphological responses in induced pluripotent stem cells (iPSCs) and astrocytes under DMSO and three pharmacological perturbations: **Latrunculin B, GW3965, and Clozapine**.

The workflow includes (1) condition-specific morphology autoencoders, (2) latent-space developmental translation from iPSCs to astrocytes (four conditions including DMSO), and (3) within-cell-type DMSO-to-drug translation (three drugs * two cell types). Evaluation includes morphology effect prediction, drug specificity, top-feature overlap, transition-vector geometry, morphology-family enrichment, and integrated-gradients attribution.

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
```

## Input data

The training notebook expects eight **preprocessed single-cell morphology .csv files** with matching feature names, one row per cell, and numeric CellProfiler-derived morphology feature columns.

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
## Suggested execution order

1. Run `notebooks/01_train_morphology_translators.ipynb` **top-to-bottom** to audit inputs, create the 70/15/15 partitions, train eight autoencoders, and run the ten specified translations. 
2. Run `notebooks/02_evaluate_drug_effects.ipynb` to evaluate perturbation recovery, feature-ranking overlap, and compound specificity.
3. Run `notebooks/03_feature_family_enrichment_and_IG.ipynb` to compute feature-family over/underrepresentation and source-feature IG enrichment. 
4. Optionally run `notebooks/04_training_and_reconstruction_plots.ipynb` for training curves and autoencoder diagnostics.
5. Generate figures using the scripts below:

```bash
python scripts/morphology_transition_vectors.py
python scripts/plot_topk_feature_recovery.py
python scripts/plot_cross_drug_specificity.py
python scripts/plot_family_enrichment.py
```
## Citation and authorship

If you use or adapt this code, please cite the associated **Cell Translator** manuscript when it becomes available. 
