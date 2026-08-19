# Acute leukemia application

This application reproduces two Cell Translator analyses from the acute
leukemia notebooks:

1. **Healthy hematopoietic differentiation:** CD34+ -> BMMC
2. **Leukemia transformation:** CD34+ -> MPAL

Both workflows use the same model/training implementation but have different
target states. The application-specific code imports shared model components
from the canonical `cell_translator` package and does not redefine the
HurdleAE or CycleGAN architectures.

## Input data

The source notebooks start from preprocessed, non-negative, log1p-normalized
expression matrices with exactly **5,000 genes in the same order**. The raw
count QC and upstream HVG-selection procedure are not contained in the supplied
model notebooks, so this public workflow intentionally does not invent one.
Provide the matrices used by the model:

- CD34+: 5,752 x 5,000
- BMMC: 6,270 x 5,000
- MPAL: 5,885 x 5,000
- an ordered 5,000-element gene-name array

All filesystem locations in the YAML files are placeholders.

## Stage 1 architecture

Unlike several other Cell Translator applications, acute leukemia trains a
**separate HurdleAE for each biological state**. The historical decoder is also
shallower than the shared HurdleAE used by the neuronal, organoid, and
pancreatic applications. The exact architecture and loss are implemented once
in:

- `cell_translator/models/independent_hurdle_ae.py`
- `cell_translator/training/independent_hurdle_ae_training.py`

The reported configuration uses a 128-dimensional latent space, batch size
256, learning rate 1e-3, up to 200 epochs, 20% validation, and seed 42.

## Stage 2 training variant

The acute-leukemia notebooks use a historical CycleGAN training schedule that
is not identical to the later default Cell Translator loop. Source and target
training latents are standardized jointly, truncated to the smaller sample
count, paired by row in one shuffled TensorFlow dataset, and trained with the
acute-leukemia-specific alignment weights. This is preserved in
`cell_translator/training/acute_leukemia_cyclegan_training.py`.

The generator/critic architectures themselves are imported from the shared
`cell_translator.models.cyclegan` module.

TensorFlow was not explicitly seeded in the source notebooks. Therefore a
fresh CycleGAN run reproduces the reported architecture, objective, data split,
and hyperparameters but is not expected to regenerate the original GAN weights
bit-for-bit. Exact numerical reproduction is strongest when the archived model
or translated output is available.

## Run

Healthy differentiation:

```bash
python applications/acute_leukemia/run.py \
  --config configs/acute_leukemia/healthy_cd34_to_bmmc.yaml \
  --mode retrain
```

Disease transformation:

```bash
python applications/acute_leukemia/run.py \
  --config configs/acute_leukemia/disease_cd34_to_mpal.yaml \
  --mode retrain
```

To evaluate an archived `translated_outputs.npz` without retraining:

```bash
python applications/acute_leukemia/run.py \
  --config configs/acute_leukemia/disease_cd34_to_mpal.yaml \
  --mode evaluate \
  --translated path/to/translated_outputs.npz
```

## Outputs retained for reproducibility

Each workflow saves expected-expression, globally hard-thresholded, and
library-size-matched translations. This is intentional because the original
notebooks used different decoded outputs for different validation blocks.
The YAML specifies which output reproduces the final per-gene distribution
analysis.

The public evaluation retains the model-performance analyses needed to support
the application:

- held-out latent mean/std agreement
- per-dimension latent KS and Wasserstein distances
- latent translation-direction cosine similarity
- global detection-rate calibration
- per-gene KS and Wasserstein distances
- final linear-space log2FC recovery

## Downstream analyses intentionally omitted

The source notebooks also contain extensive latent-axis/Jacobian sensitivity,
marker-category trajectories, kNN biological-state comparisons, healthy versus
disease driver comparisons, and disease-deviation/category analyses. These are
post-hoc biological explorations rather than requirements for reproducing the
core Cell Translator translation results, so they are not part of the main
public execution path.

The notebooks also contain Integrated Gradients/DriverScore analyses. They are
not included in this minimal application package because the later DriverScore
combines several of the omitted axis/sensitivity analyses. Including only part
of that chain would not faithfully reproduce the reported downstream score.

## Reference values

`configs/acute_leukemia/expected_results.yaml` records the main values printed
by the source notebooks. Use `validate_results.py` after running/evaluating a
workflow to compare reproduced values to those references.
