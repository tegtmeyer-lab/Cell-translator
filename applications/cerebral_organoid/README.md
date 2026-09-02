# Cerebral organoid maturation

This application evaluates Cell Translator on three independent scRNA-seq cerebral-organoid maturation datasets. The HurdleAE and latent WGAN-GP CycleGAN are imported from the shared `cell_translator` package; this directory contains only organoid-specific preparation, orchestration, evaluation, and interpretation code.

## Datasets

The cerebral organoid analyses use three publicly available single-cell RNA-seq datasets:

| Dataset | Translation | Data access |
|---|---|---|
| Uzquiano et al. cortical organoid atlas | 2 months --> 6 months | [Broad Single Cell Portal — SCP1756](https://singlecell.broadinstitute.org/single_cell/study/SCP1756) |
| He et al. Human Neural Organoid Cell Atlas (HNOCA) | 15 days --> 60 days | [HNOCA cleaned datasets — Zenodo](https://zenodo.org/records/14161275) |
| Fleck et al. brain organoid developmental time course | 18 days --> 61 days | [Processed data — Zenodo](https://zenodo.org/records/5242913) / [Raw scRNA-seq — ArrayExpress E-MTAB-12001](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-12001) |

Download the corresponding
dataset from the links above and update the local input path in the relevant
configuration file before running the application.

For the primary dataset, HVGs are selected jointly across the 1-, 2-, 3-, 4-, 5-, and 6-month samples. For HNOCA/He, exact ages 15, 30, 60, 90, 120, 150, and 180 days are used for HVG selection. For the third dataset, exact ages 4, 9, 12, 18, 21, and 61 days are used.

The supplied matrices are already normalized/log1p. No additional normalization or final QC filtering is applied before HVG selection. HVGs are selected with Scanpy `flavor="seurat_v3"` and `batch_key="timepoint"`. Scanpy can therefore emit a warning that Seurat-v3 HVG selection expects raw counts.

## Prepare a dataset

Replace only the placeholder paths in the selected YAML, then run:

```bash
python applications/cerebral_organoid/prepare_data.py \
  --config configs/cerebral_organoid/primary_scp1756.yaml
```

## Full retraining workflow

```bash
python applications/cerebral_organoid/run.py \
  --config configs/cerebral_organoid/primary_scp1756.yaml \
  --all \
  --no-show-plots
```

The same runner works for the two independent datasets by changing the YAML path.

## Evaluate an archived translated matrix

A CycleGAN training run therefore reproduces the training procedure:

```bash
python applications/cerebral_organoid/run.py \
  --config configs/cerebral_organoid/primary_scp1756.yaml \
  --evaluate \
  --translated path/to/translated_6month_genespace_hvg.npy
```

This requires the corresponding Stage-1 AE output file so the same held-out target cells can be recovered.

## Primary-dataset validation

The final validation workflow compares translated held-out 2-month cells with held-out real 6-month cells. Reference values recorded in `configs/cerebral_organoid/expected_results.yaml`, including:

- gene-mean Pearson r = 0.9583
- detection-rate Pearson r = 0.9481
- median per-gene KS = approximately 0.0011
- mean-log1p effect correlation r = 0.9908

Validate reproduced metrics with:

```bash
python applications/cerebral_organoid/validate_results.py \
  --metrics path/to/results/primary_organoid/evaluation_metrics.json \
  --expected configs/cerebral_organoid/expected_results.yaml
```

## Driver-gene analysis

The final coherent 128-dimensional Integrated Gradients / DriverScore workflow is retained for the primary dataset:

```bash
python applications/cerebral_organoid/interpret.py \
  --config configs/cerebral_organoid/primary_scp1756.yaml
```

It uses 64 Integrated Gradients steps, at most 1,024 cells, a mean-expression baseline, batch size 16, and seed 42. The final detection filter requires a gene to be detected in at least 10% of cells in both source and target states, followed by 75th-percentile DriverScore/IG categorization.

