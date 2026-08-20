# Neuronal differentiation

This application uses Cell Translator to model neuronal differentiation from preprocessed single-cell RNA-seq expression data.

## Data preprocessing

The single-cell RNA-seq data were preprocessed using Scanpy following its standard preprocessing workflow, including quality control, normalization, log transformation, and gene filtering/selection.

See the [Scanpy preprocessing and clustering tutorial](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html) for the preprocessing workflow used.

## Configuration

Application-specific input paths, source/target state definitions, model parameters, and output locations are specified in:

```text
config.yaml
