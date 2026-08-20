# Perturbation tool benchmarks: scGen, CellOT, and scGPT

This module prepares the external perturbation-model outputs used in the comparison with Cell Translator across **APOE, MSI2, NR1H3, LRRC4C, and ANKRD12**. It does not redefine or modify Cell Translator.

All methods are aligned to the same benchmark gene ordering (18,614 genes) and, to the same **2,106 control cells** used by the CellOT held-out split.

## Source-derived benchmark setup

The combined benchmark AnnData contains 21,971 cells and 18,614 genes: 10,529 control, 8,828 LRRC4C, 684 MSI2, 681 APOE, 681 NR1H3, and 568 ANKRD12 cells.

### CellOT

CellOT was implemented following the official
[CellOT repository](https://github.com/bunnech/cellot).

The workflow selects 2,000 highly variable genes (HVGs) from the **control
cells only** using Scanpy (`flavor="seurat"`). PCA with 50 components is then
fit on the control-cell HVG expression matrix (`random_state=0`) and used to
transform all conditions. CellOT is trained on this 50-dimensional
representation.

The CellOT task configuration uses a batch size of 256, a
condition-stratified 80:20 train/test split (`test_size=0.20`,
`random_state=0`), and 100,000 training iterations. Evaluation is performed
with `setting=iid` and `where=data_space`.

Following CellOT evaluation, predicted PCA coordinates are inverse-transformed
to the 2,000-HVG expression space and clipped at zero. Predictions are then
expanded to the complete 18,614-gene expression space; genes outside the HVG
set are assigned their measured control-state mean expression.

The resulting 2,106 held-out control-cell observation IDs define the common
evaluation set and are reused for alignment of the scGen and scGPT
predictions.

### scGen

scGen was implemented following the official
[scGen repository](https://github.com/theislab/scgen).

The benchmark environment used Python 3.10.19, scGen 2.1.1, and
scvi-tools 1.3.3. The complete preprocessed expression matrix containing
control cells and the five perturbation conditions is used for model
training.

The condition label is registered using

`SCGEN.setup_anndata(adata, batch_key="condition")`

followed by construction of `SCGEN(adata)` and training for 50 epochs.

For each perturbation, `SCGEN.predict()` is applied to the same 2,106
held-out control cells defined by the CellOT test set. Predictions are
therefore generated for an identical set of source cells across CellOT,
scGen, and scGPT for all five perturbations: APOE, MSI2, NR1H3, LRRC4C,
and ANKRD12.

### scGPT

scGPT was implemented following the official
[scGPT repository](https://github.com/bowang-lab/scGPT) and its
perturbation-prediction workflow.

The benchmark environment used scGPT 0.2.4, PyTorch 2.3.1+cu118, and
torchtext 0.18.0+cpu. Perturbation data are organized using the GEARS
`PertData` representation, with each cell represented by its expression
profile and a binary perturbation indicator.

The pretrained scGPT brain checkpoint is used to initialize the model.
Checkpoint parameters corresponding to the gene encoder, value encoder, and
transformer encoder are loaded, including conversion of fused attention
weights when required by the PyTorch transformer implementation. These
pretrained backbone components are then frozen, while the remaining model
parameters are fine-tuned on the perturbation data.

Fine-tuning uses seed 42, a maximum sequence length of 256 genes, Adam
optimization with a learning rate of `1e-3`, masked mean-squared-error loss,
gradient clipping at 1.0, and two training epochs. During training, up to
256 genes are sampled per batch. During inference, all 18,614 genes are
predicted in sequential 256-gene chunks.

For cross-method evaluation, the CellOT held-out control-cell IDs are matched
to the corresponding scGPT/GEARS control graphs. Each of the five
perturbations is then applied to the same 2,106 control cells using a
single-gene perturbation mask, producing one prediction matrix per
perturbation in the common 18,614-gene space.

For the benchmark expression matrices, the exported scGPT `yhat` values are
converted to the absolute expression scale used by the other methods by

`X_pred = mean(control expression) + yhat`

followed by clipping negative values to zero.

## Commands

Edit `configs/benchmarks/perturbation_tools.yaml` to point to local files.

CellOT preparation:

```bash
python -m analyses.perturbation_tool_benchmarks.cellot_prepare \
  --config configs/benchmarks/perturbation_tools.yaml
```

Run CellOT from the environment containing the external CellOT checkout:

```bash
python -m analyses.perturbation_tool_benchmarks.cellot_train \
  --config configs/benchmarks/perturbation_tools.yaml
```

Export CellOT predictions and the common held-out control IDs:

```bash
python -m analyses.perturbation_tool_benchmarks.cellot_export \
  --config configs/benchmarks/perturbation_tools.yaml
```

Run scGen after the common ID file exists:

```bash
python -m analyses.perturbation_tool_benchmarks.scgen_runner \
  --config configs/benchmarks/perturbation_tools.yaml
```

Postprocess archived scGPT raw predictions:

```bash
python -m analyses.perturbation_tool_benchmarks.scgpt_postprocess \
  --config configs/benchmarks/perturbation_tools.yaml
```

The final cross-model metrics are already centralized in:

```bash
python applications/genetic_perturbation/compare_models.py \
  --config configs/genetic_perturbation/comparison.yaml
```

After updating that comparison YAML to point at the generated scGen, CellOT, scGPT, and Cell Translator arrays, validate the benchmark rows with:

```bash
python -m analyses.perturbation_tool_benchmarks.validate_comparison \
  --csv path/to/model_comparison_summary.csv \
  --expected configs/benchmarks/perturbation_tools_expected.yaml
```

## Environment note

These external tools have sensitive dependency stacks. They should be run in **separate environments** rather than adding scGen, CellOT, scGPT/GEARS, and their version-specific PyTorch dependencies to the main Cell Translator environment.
