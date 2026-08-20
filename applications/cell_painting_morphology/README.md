# Cell Painting morphology translation

This application reproduces the dense-morphology branch of Cell Translator.
Unlike the six scRNA-seq applications, it does **not** use a HurdleAE. Cell
Painting features are continuous and are modeled with state-specific standard
autoencoders followed by a morphology-specific latent WGAN-GP CycleGAN.

## Verified workflow represented here

The coherent source-notebook workflow uses annotated Cell Painting profiles,
3,409 shared non-technical morphology features, 256-dimensional state-specific
AEs, and held-out source-cell translation.

Two configurations are provided:

- `control_line24.yaml`: control line 24, Stem→NPC, NPC→Neuron, NPC→Astrocyte.
- `deletion_line45.yaml`: 22q11.2-deletion line 45, Stem→NPC and
  NPC→Astrocyte.

The deletion NPC→Neuron path is intentionally not exposed. Although deletion
neuron data were created in the notebook, the later AE dictionary referenced
the earlier control-neuron variable. The printed deletion AE summary contains
2,828 neuron cells—the control-neuron count—rather than either deletion-neuron
count prepared immediately beforehand. Reproducing that path would reproduce a
notebook-state error rather than a valid deletion analysis.

## Data preparation

Tracked configs contain placeholders only. Set the four control parquet paths
(or the three verified deletion paths), then run:

```bash
python applications/cell_painting_morphology/run.py \
  --config configs/cell_painting_morphology/control_line24.yaml \
  --prepare
```

Preparation:

1. selects the exact line ID and genotype;
2. intersects columns across the requested states;
3. removes metadata/technical leakage columns with the notebook's exclusion
   patterns;
4. keeps numeric morphology columns present in every state; and
5. writes ordered feature names plus per-state NumPy matrices.

For deletion, the config points to the verified control 3,409-feature list.
This avoids reintroducing the contaminated neuron state solely for feature-list
construction; the notebook later confirms the control and deletion feature
lists were both 3,409 features.

## Train the state-specific morphology AEs

```bash
python applications/cell_painting_morphology/run.py \
  --config configs/cell_painting_morphology/control_line24.yaml \
  --train-ae
```

Each state is split 80:20 with `random_state=42`. Missing values are filled
using the **training-state median**, and `StandardScaler` is fitted on training
cells only. The AE training objective is MSE + variance matching + covariance
penalty; validation/early stopping use MSE only, exactly as in the notebook.

The notebook did not explicitly seed PyTorch before AE initialization or data
loader shuffling. Consequently, a fresh training run reproduces the method and
split but cannot be promised to reproduce the exact archived weights. Archived
AE checkpoints are the strongest route to exact numerical reproduction.

## Train and evaluate translations

Run all enabled transitions:

```bash
python applications/cell_painting_morphology/run.py \
  --config configs/cell_painting_morphology/control_line24.yaml \
  --train-gan --evaluate
```

Or select one transition by its exact YAML label:

```bash
python applications/cell_painting_morphology/run.py \
  --config configs/cell_painting_morphology/control_line24.yaml \
  --transition "NPC → Astrocyte" \
  --train-gan --evaluate
```

The corrected validation calculates feature-mean correlation, feature-SD
correlation, per-feature KS statistics, and target-neighbor percentage against
a reference containing **both measured source and measured target cells**.
Earlier notebook cells that fit a nearest-neighbor model on target cells alone
necessarily returned approximately 100% target purity and are not used as the
reference validation.

## Control/deletion effect-size comparison

After both valid workflows have translated outputs:

```bash
python applications/cell_painting_morphology/effect_size_comparison.py \
  --control-config configs/cell_painting_morphology/control_line24.yaml \
  --deletion-config configs/cell_painting_morphology/deletion_line45.yaml
```

This reproduces the notebook's feature-wise Cohen's d comparison after removing
neighbor features, leaving 3,394 shared features. Because deletion NPC→Neuron
is excluded, the public comparison contains the two verified transitions:
Stem→NPC and NPC→Astrocyte.

## Optional Integrated Gradients / DriverScore

```bash
python applications/cell_painting_morphology/interpret.py \
  --config configs/cell_painting_morphology/control_line24.yaml \
  --transition "Stem → NPC"
```

This downstream module preserves the historical Keras→PyTorch generator mirror
used for IG and prints the mirror's maximum forward discrepancy before
computing attributions. It then writes `integrated_gradients.csv` and
`driver_features.csv`.

## Reference results

Notebook reference values are stored in
`configs/cell_painting_morphology/expected_results.yaml`, including the four
control AE reconstruction summaries, the three corrected control translations,
the three verified deletion AEs, the two valid deletion translations, and the
control/deletion effect-size correlations.

## Source-notebook sections intentionally not promoted to the main workflow

The notebook also contains later INT-normalized Stem→Astrocyte experiments with
different feature counts (for example 1,696 and 1,662), a 150-epoch control AE,
and an "all individual" section that reads externally subsampled files not
created by the notebook. Some of those later cells also reuse stale target/state
variables. They are not merged into this 3,409-feature reproducibility pipeline.
If those exact outputs are reported independently, they should be organized
from the clean source artifacts/notebook that generated them rather than inferred
from this notebook state.
