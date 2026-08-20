# Cell Painting morphology translation

This application reproduces the dense-morphology branch of Cell Translator.
Unlike the six scRNA-seq applications, it does **not** use a HurdleAE. Cell
Painting features are continuous and are modeled with state-specific standard
autoencoders followed by a morphology-specific latent WGAN-GP CycleGAN.

## Verified workflow represented here

The coherent workflow uses annotated Cell Painting profiles,
3,409 shared non-technical morphology features, 256-dimensional state-specific
AEs, and held-out source-cell translation.

Two configurations are provided:

- `control_line24.yaml`: control line 24, Stem→NPC, NPC→Neuron, NPC→Astrocyte.
- `deletion_line45.yaml`: 22q11.2-deletion line 45, Stem→NPC and
  NPC→Astrocyte.

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
construction.

## Train the state-specific morphology AEs

```bash
python applications/cell_painting_morphology/run.py \
  --config configs/cell_painting_morphology/control_line24.yaml \
  --train-ae
```

Each state is split 80:20 with `random_state=42`. Missing values are filled
using the **training-state median**, and `StandardScaler` is fitted on training
cells only. The AE training objective is MSE + variance matching + covariance
penalty; validation/early stopping use MSE only.


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

The validation calculates feature-mean correlation, feature-SD
correlation, per-feature KS statistics, and target-neighbor percentage against
a reference containing **both measured source and measured target cells**.

## Control/deletion effect-size comparison

After workflows have translated outputs:

```bash
python applications/cell_painting_morphology/effect_size_comparison.py \
  --control-config configs/cell_painting_morphology/control_line24.yaml \
  --deletion-config configs/cell_painting_morphology/deletion_line45.yaml
```

This reproduces the feature-wise Cohen's d comparison after removing
neighbor features, leaving 3,394 shared features. 

## Optional Integrated Gradients / DriverScore

```bash
python applications/cell_painting_morphology/interpret.py \
  --config configs/cell_painting_morphology/control_line24.yaml \
  --transition "Stem → NPC"
```

## Reference results

reference values are stored in
`configs/cell_painting_morphology/expected_results.yaml`, including the four
control AE reconstruction summaries, the three corrected control translations,
the three verified deletion AEs, the two valid deletion translations, and the
control/deletion effect-size correlations.

