# Cell Translator

Cell Translator is a HurdleAE + latent-space CycleGAN framework for translating
single-cell states. The repository uses one shared implementation of the model
components and keeps biological preprocessing, source/target definitions,
evaluation, and validation within independent application directories.

## Repository design

**One canonical method implementation; independent application pipelines.**

Shared code lives under `cell_translator/`:

```text
cell_translator/
├── models/
│   ├── hurdle_ae.py
│   ├── perturbation_hurdle_ae.py
│   └── cyclegan.py
├── training/
│   ├── hurdle_ae_training.py
│   ├── perturbation_hurdle_ae_training.py
│   └── cyclegan_training.py
├── inference/
│   ├── translate.py
│   └── perturbation.py
└── utils/
```

Application scripts import these implementations rather than redefining model
classes or training losses.

## Install

```bash
pip install -e .
```

The source notebooks did not record all exact package-version numbers. The
requirements therefore list required packages without claiming unavailable
version provenance. If the original environment file is available, pin those
versions before archival release.

## Applications currently organized

### Neuronal differentiation

```bash
python applications/neuronal_differentiation/run.py \
  --config applications/neuronal_differentiation/config.yaml \
  --translation all \
  --train-ae \
  --no-show-plots
```

### Genetic perturbation

The perturbation application covers control→APOE, MSI2, NR1H3, LRRC4C, and
ANKRD12 translation. It preserves the separate-control/target-encoder HurdleAE
architecture used in that analysis while sharing the common CycleGAN model and
loss implementation.

See [`applications/genetic_perturbation/README.md`](applications/genetic_perturbation/README.md)
for preparation, archived-model reproduction, retraining, evaluation,
interpretation, baseline comparison, and result-validation commands.


### Cerebral organoid maturation

The organoid application contains one shared maturation workflow with three
dataset-specific preparation/configuration files.

See [`applications/cerebral_organoid/README.md`](applications/cerebral_organoid/README.md).

### Pancreatic endocrine differentiation

The pancreatic application reproduces NGN3-high endocrine progenitor →
Endocrine translation in GSE132188 using the final 5,003-gene input.

See [`applications/pancreatic_endocrine/README.md`](applications/pancreatic_endocrine/README.md).

## Paths and data

Tracked YAML files contain placeholder paths only. No machine-specific,
user-specific, or compute-environment paths are required by the source code.
Processed matrices are expected to preserve the exact gene order described by
each application.
