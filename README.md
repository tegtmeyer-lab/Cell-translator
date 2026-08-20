# Cell Translator

Cell Translator is a computational framework for predicting cellular state transitions from single-cell transcriptomic and Cell Painting morphology profiles. The repository provides reusable implementations of the Cell Translator model components while preserving application-specific model configurations, biological preprocessing, source and target state definitions, evaluation, and validation workflows.

## Repository design

**A shared Cell Translator framework with independent application-specific pipelines.**

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

## Data preprocessing

Raw single-cell RNA-seq data were processed using Scanpy following its standard single-cell preprocessing workflow. Application-specific quality-control filters, normalization, gene selection, and output formatting used to generate the Cell Translator input matrices are documented in the corresponding application directories.

See the [Scanpy preprocessing and clustering tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html) for the general preprocessing workflow.

## Install

Clone the repository and install Cell Translator in editable mode:

```bash
git clone <repository-url>
cd Cell-Translator
pip install -e .
```

For full reproducibility, we recommend using the package versions specified in the provided environment or requirements files.

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

The pancreatic application reproduces NGN3-high endocrine progenitor -->
Endocrine translation in GSE132188 using the final 5,003-gene input.

See [`applications/pancreatic_endocrine/README.md`](applications/pancreatic_endocrine/README.md).

### Acute leukemia

The acute leukemia application contains two workflows: healthy CD34+ --> BMMC
differentiation and CD34+ --> MPAL leukemia transformation. It preserves the
independent-state HurdleAE and historical acute-leukemia CycleGAN training
variant used in the source analyses.

See [`applications/acute_leukemia/README.md`](applications/acute_leukemia/README.md).

### Cell Painting morphology

The morphology application uses state-specific dense autoencoders and a
morphology-specific latent CycleGAN for Cell Painting profiles. The organized
workflow covers the coherent 3,409-feature control-line translations and the
verified deletion Stem->NPC and NPC->Astrocyte translations.

See [`applications/cell_painting_morphology/README.md`](applications/cell_painting_morphology/README.md).

## Paths and data

Tracked YAML files contain placeholder paths only. No machine-specific,
user-specific, or compute-environment paths are required by the source code.
Processed matrices are expected to preserve the exact gene order described by
each application.

## Validation and benchmarking analyses

Validation and benchmarking code that is not itself a biological application lives under `analyses/`.

### Perturbation baseline comparison

Compares archived Cell Translator predictions with Identity, MeanShift,
NearestNeighbor, and LinearLatentShift baselines using perturbation-effect,
distributional, and discriminator metrics.

See [`analyses/baseline_comparison/README.md`](analyses/baseline_comparison/README.md).

### Leave-one-line-out morphology validation

Evaluates generalization to entirely held-out control cell lines across five
neural differentiation transitions.

See [`analyses/leave_one_line_out/README.md`](analyses/leave_one_line_out/README.md).
### Perturbation tool benchmarks

Prepares the scGen, CellOT, and archived-scGPT benchmark outputs used in the
five-perturbation comparison with Cell Translator. External tool code remains
separate from the Cell Translator implementation.

See [`analyses/perturbation_tool_benchmarks/README.md`](analyses/perturbation_tool_benchmarks/README.md).

