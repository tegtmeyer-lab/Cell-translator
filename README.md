# Cell Translator

Cell Translator is a state-translation framework spanning sparse single-cell
transcriptomics and continuous Cell Painting morphology. The repository uses one shared implementation of the model
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
│   ├── independent_hurdle_ae.py
│   ├── morphology_ae.py
│   ├── morphology_cyclegan.py
│   └── cyclegan.py
├── training/
│   ├── hurdle_ae_training.py
│   ├── perturbation_hurdle_ae_training.py
│   ├── independent_hurdle_ae_training.py
│   ├── acute_leukemia_cyclegan_training.py
│   ├── morphology_ae_training.py
│   ├── morphology_cyclegan_training.py
│   └── cyclegan_training.py
├── inference/
│   ├── translate.py
│   ├── perturbation.py
│   └── morphology.py
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

### Acute leukemia

The acute leukemia application contains two workflows: healthy CD34+ -> BMMC
differentiation and CD34+ -> MPAL leukemia transformation. It preserves the
independent-state HurdleAE and historical acute-leukemia CycleGAN training
variant used in the source analyses.

See [`applications/acute_leukemia/README.md`](applications/acute_leukemia/README.md).

### Cell Painting morphology

The morphology application uses state-specific dense autoencoders and a
morphology-specific latent CycleGAN for Cell Painting profiles. The organized
workflow covers the coherent 3,409-feature control-line translations and the
verified deletion Stem→NPC and NPC→Astrocyte translations.

See [`applications/cell_painting_morphology/README.md`](applications/cell_painting_morphology/README.md).

## Paths and data

Tracked YAML files contain placeholder paths only. No machine-specific,
user-specific, or compute-environment paths are required by the source code.
Processed matrices are expected to preserve the exact gene order described by
each application.

## Validation and benchmarking analyses

Reviewer-facing validation and benchmarking code that is not itself a biological application lives under `analyses/`.

### Perturbation baseline comparison

Compares archived Cell Translator predictions with Identity, MeanShift,
NearestNeighbor, and LinearLatentShift baselines using perturbation-effect,
distributional, and discriminator metrics.

See [`analyses/baseline_comparison/README.md`](analyses/baseline_comparison/README.md).

### Leave-one-line-out morphology validation

Evaluates generalization to entirely held-out control cell lines across five
neural differentiation transitions. The historically used reduced validation
architecture is kept local to this analysis and does not replace the canonical
morphology implementation.

See [`analyses/leave_one_line_out/README.md`](analyses/leave_one_line_out/README.md).
