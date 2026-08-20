# Pancreatic endocrine differentiation

This application reproduces Cell Translator analysis of **NGN3-high endocrine
progenitor → Endocrine** maturation in the mouse pancreatic development atlas
GSE132188.

## Final analysis path

- source: `Ngn3 high EP` cells at day ≤ 14.5 (1,861 cells)
- target: `Endocrine` cells at day ≥ 14.5 (2,621 cells)
- input atlas: already normalized/log1p in the source analysis
- HVG selection: 5,000 genes using Scanpy `flavor="seurat"` on the combined
  source/target subset
- force-added markers: `Gcg`, `Chgb`, and `Kcnj11`
- final model input: 5,003 genes
- HurdleAE latent dimension: 64
- translation/evaluation: held-out source cells (372 translated cells) against
  held-out target cells (525 measured cells)


## Prepare data

Set placeholder paths in:

```text
configs/pancreatic_endocrine/gse132188_ngn3_to_endocrine.yaml
```

Then run:

```bash
python applications/pancreatic_endocrine/prepare_data.py   --config configs/pancreatic_endocrine/gse132188_ngn3_to_endocrine.yaml
```

## Train and evaluate

```bash
python applications/pancreatic_endocrine/run.py   --config configs/pancreatic_endocrine/gse132188_ngn3_to_endocrine.yaml   --all   --no-show-plots
```


## Fold-change and interpretability

```bash
python applications/pancreatic_endocrine/log2fc_analysis.py   --config configs/pancreatic_endocrine/gse132188_ngn3_to_endocrine.yaml

python applications/pancreatic_endocrine/interpret.py   --config configs/pancreatic_endocrine/gse132188_ngn3_to_endocrine.yaml
```

Optional projection onto the atlas UMAP:

```bash
python applications/pancreatic_endocrine/atlas_projection.py   --config configs/pancreatic_endocrine/gse132188_ngn3_to_endocrine.yaml
```

## Reference-result validation

```bash
python applications/pancreatic_endocrine/validate_results.py   --metrics path/to/results/pancreatic/evaluation_metrics.json   --expected configs/pancreatic_endocrine/expected_results.yaml
```

The reference values in `expected_results.yaml` were transcribed from the
executed outputs and are intended as reproducibility checks.
