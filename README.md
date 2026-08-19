# Cell Translator

Reviewer-facing implementation of the shared Cell Translator engine used across
the manuscript's scRNA-seq applications. The reusable HurdleAE, latent-space
CycleGAN, training routines, and inference/decode logic are defined once under
`cell_translator/`. Biological applications import this core package and retain
their own preprocessing, source/target definitions, evaluation, and validation.

## Design principle

**One canonical method implementation; independent application pipelines.**

Do not copy or redefine HurdleAE, CycleGAN, or the shared training functions
inside `applications/`.

## Install

```bash
pip install -e .
```

Before the archival/reviewer release, replace unpinned dependencies in
`requirements.txt` with the exact software versions used for the manuscript.

## Current application runner

The neuronal differentiation application has been refactored from the
reviewer-ready reference script and can be invoked as:

```bash
python applications/neuronal_differentiation/run.py \
  --config applications/neuronal_differentiation/config.yaml \
  --translation all \
  --train-ae \
  --no-show-plots
```

The YAML contains placeholder paths only. Replace them with the local paths to
the manuscript input matrices and the ordered HVG file.

## Remaining applications

Their application-specific code should be added only after checking the exact
source analysis for preprocessing, state definitions, evaluation, and validation.
The shared method should continue to be imported from `cell_translator`.

## Hyperparameter ownership

Shared training constants are defined once in `cell_translator/hyperparameters.py`.
Application-specific model dimensions remain in the application YAML. For the
neuronal application, the preserved values are 64 dimensions for the four-state
HurdleAE and 128 dimensions for the NPC/Astrocyte HurdleAE.
