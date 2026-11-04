# Methods and interpretation

## Experiment 2: Within-condition developmental translation

Six independent NPC→Neuron models are fit, one each for control, APOE, MSI2, NR1H3, ANKRD12 and LRRC4C. For each state/condition, cells are split 70/15/15 for training, validation and test. Train-only AE thresholds and latent standardization are used. A condition-specific HurdleAE and latent CycleGAN learn the state translation. Final metrics are computed on held-out test populations.

## Experiment 3: Leave-one-perturbation-out (LOPO)

For each of five perturbations, independently fit a pooled developmental model using control plus four other perturbations. Neither NPC nor Neuron examples from the withheld condition are used for fitting or validation. Held-out NPC test cells, frozen in Experiment 2, are translated after training; held-out Neuron cells are used only for evaluation.

## Data and leakage caveats

- The Experiment 2 default split is cell-level, not donor-level. Provide biological group labels if available.
- LOPO excludes target-condition cells but does not exclude shared donors or batches across other conditions.
- Genes are intersected across all 12 input matrices in Experiment 2. This uses held-out *feature names* in Experiment 3; it is a transductive gene-universe definition, not an inductively selected feature space.
- Input files must have one row per cell, one column per gene, with barcode/index in the first column. Reused barcodes are allowed and positional IDs are assigned.
- There is no cell-to-cell pairing; metrics compare population distributions or gene-wise population summaries.
- Top-100 developmental Jaccard means ranking by absolute mean change, **not** statistically significant DEGs.
- `gene_mean_mae` and `delta_mae` are mathematically equal when both use the same NPC reference. Do not present these as independent findings.
- Expression inputs assumed nonnegative and on a consistent scale (e.g., log1p(CP10K)).
- For publication, confirm sample/replicate structure, evaluate uncertainty, and confirm test-cell consistency.
