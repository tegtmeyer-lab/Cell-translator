# Perturbation tool benchmarks: scGen, CellOT, and scGPT

This module prepares the external perturbation-model outputs used in the comparison with Cell Translator across **APOE, MSI2, NR1H3, LRRC4C, and ANKRD12**. It does not redefine or modify Cell Translator.

All methods are aligned to the same benchmark gene ordering (18,614 genes) and, where the source notebooks support it, to the same **2,106 control cells** used by the CellOT held-out split.

## Source-derived benchmark setup

The combined benchmark AnnData contains 21,971 cells and 18,614 genes: 10,529 control, 8,828 LRRC4C, 684 MSI2, 681 APOE, 681 NR1H3, and 568 ANKRD12 cells.

### CellOT

The final notebook workflow selects 2,000 HVGs on **control cells only** using Scanpy `flavor="seurat"`, fits PCA(50) on those control HVGs (`random_state=0`), transforms all conditions, and trains CellOT on the 50-dimensional representation. The CellOT task configuration uses batch size 256, a condition-stratified 80:20 split (`random_state=0`), and 100,000 training iterations. Evaluation uses `setting=iid` and `where=data_space`.

After CellOT evaluation, PCA predictions are inverse-transformed to the 2,000-HVG expression space, clipped at zero, and expanded back to all 18,614 genes. **Non-HVG genes are filled with the measured control mean**, exactly as in the source notebook. The common 2,106 held-out control observation IDs are exported from CellOT and reused by scGen/scGPT alignment.

The supplied notebook does **not** record the exact CellOT package version or repository commit; it calls an external `configs/models/cellot.yaml`. For exact historical retraining, use the original CellOT checkout if it is available. The preparation and postprocessing steps in this directory are fully specified.

### scGen

The recorded environment was Python 3.10.19, scGen 2.1.1, and scvi-tools 1.3.3. The notebook calls `SCGEN.setup_anndata(..., batch_key="condition")`, constructs `SCGEN(adata)`, and trains for 50 epochs. Predictions are generated for the exact 2,106 CellOT test-control IDs for all five perturbations.

No random seed was explicitly set in the source notebook, so the public configuration leaves the seed `null` rather than adding one retrospectively.

### scGPT

The supplied scGPT notebook is **not a complete from-scratch training recipe**. It records scGPT 0.2.4, PyTorch 2.3.1+cu118, and torchtext 0.18.0+cpu; loads a brain pretrained checkpoint into selected encoder prefixes; and constructs perturbation predictions on the CellOT-matched control cells. However, the notebook references `max_seq_len` and `map_raw_id_to_vocab_id` without defining them, leaves 23 parameter tensors outside the loaded checkpoint prefixes, and contains no subsequent fine-tuning step that would resolve their state. Exact raw scGPT prediction generation therefore cannot be reconstructed faithfully from these files alone.

For this reason, the reproducible public path starts from the **archived raw scGPT `.npz` predictions** and reproduces the final postprocessing used in the comparison:

`absolute prediction = mean(control expression) + yhat`, followed by clipping at zero.

This is intentionally separated from any speculative from-scratch scGPT implementation.

## Commands

Edit `configs/benchmarks/perturbation_tools.yaml` to point to local files.

CellOT preparation:

```bash
python -m analyses.perturbation_tool_benchmarks.cellot_prepare \
  --config configs/benchmarks/perturbation_tools.yaml
```

Run CellOT from the environment containing the historical/external CellOT checkout:

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

These external tools have mutually sensitive dependency stacks. They should be run in **separate environments** rather than adding scGen, CellOT, scGPT/GEARS, and their version-specific PyTorch dependencies to the main Cell Translator environment.
