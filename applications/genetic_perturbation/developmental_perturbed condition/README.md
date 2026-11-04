# Cell Translator — Developmental Translation and Unseen-Perturbation Generalization

Research implementation of two complementary experiments using a Hurdle Autoencoder (HurdleAE) and latent-space CycleGAN. This repository contains **code only**, not expression matrices, model checkpoints or identifiable data.

> **Citation:** If you use or adapt this code or approach, please cite the associated Cell Translator manuscript once its bibliographic details are available. Replace this sentence with the formal paper citation and DOI on publication.

## Research questions

**Experiment 2 — Developmental prediction within known conditions:** independently learn NPC → Neuron translation for control, APOE, MSI2, NR1H3, ANKRD12 and LRRC4C. This is a within-condition held-out-cell evaluation; the target state is represented in model training.

**Experiment 3 — Leave-one-perturbation-out (LOPO):** train a new pooled NPC → Neuron model on control and four perturbed conditions, excluding **both NPC and Neuron cells** of the fifth perturbation. Translate that perturbation's frozen NPC test population and compare against its frozen measured Neuron test population. Repeat for all five holdouts. This evaluates cross-perturbation transfer, not genotype-conditioned prediction.

## Repository layout

```text
cell_translator_developmental_github/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.example.json
├── scripts/
│   ├── experiment2_developmental.py
│   └── experiment3_lopo.py
├── notebooks/
│   ├── experiment2_developmental.ipynb
│   └── experiment3_lopo.ipynb
└── docs/
    └── METHODS.md
```

## Installation

Requires a Python environment with compatible **PyTorch and TensorFlow/Keras** installations. Dependencies are listed in `requirements.txt`; GPU-specific installation of PyTorch/TensorFlow may need adjustment for your compute environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/config.example.json configs/config.json
```

Edit `configs/config.json` to specify input CSVs and output directory. Paths are relative to the repository root unless given as absolute paths. **Do not commit `config.json` containing private paths**; `.gitignore` excludes it.

## Input layout and assumptions

The example configuration expects input matrices like:

```text
data/
├── control/
│   ├── npc_log1p_cp10k.csv
│   └── neuron_log1p_cp10k.csv
├── APOE/
│   ├── npc_log1p_cp10k.csv
│   └── neuron_log1p_cp10k.csv
└── ... (MSI2, NR1H3, ANKRD12, LRRC4C)
```

Each CSV contains cells as rows, genes as named columns, and an index/barcode in the first column. Barcodes need not be unique; the loader assigns unique positional IDs. Inputs should share preprocessing and be nonnegative on the same expression scale (for example, `log1p(CP10K)`). Gene order is aligned by the Experiment 2 code.

The six conditions require **12 CSVs**. If biological replicate/donor labels are available, specify group-label CSVs for Experiment 2 in `group_files`; otherwise it uses cell-level splits, which are less stringent.

## Running Experiment 2

Start with one condition:

```bash
python scripts/experiment2_developmental.py --config configs/config.json --condition APOE
```

Then, after reviewing train/validation/test counts and training histories, run all six:

```bash
python scripts/experiment2_developmental.py --config configs/config.json --all
```

Aggregate completed experiments without retraining:

```bash
python scripts/experiment2_developmental.py --config configs/config.json --aggregate-only
```

Results appear under `outputs/developmental/<condition>_NPC_to_Neuron/` by default. Saved artifacts include HurdleAE checkpoint, trained generators, latent spaces, translated test expression, split assignments, metrics and manifest. `all_conditions_metrics_summary.csv` aggregates available results.

## Running Experiment 3

**Run Experiment 2 first.** Experiment 3 requires its common `gene_names.npy` and frozen test NPZ files so the held-out and within-condition models are evaluated on the same populations.

Check one fold:

```bash
python scripts/experiment3_lopo.py --config configs/config.json --holdout APOE
```

Run all five independent holdout folds:

```bash
python scripts/experiment3_lopo.py --config configs/config.json --all
```

Aggregate completed folds without retraining:

```bash
python scripts/experiment3_lopo.py --config configs/config.json --aggregate-only
```

Each fold saves its checkpoints, translated Neuron test array, latent data, training provenance, per-gene results and metric summaries in `outputs/developmental/leave_one_perturbation_out/heldout_<condition>/`. The aggregate files are `generalization_summary.csv` and `generalization_vs_within_condition.csv`.

## Notebook interface

The two notebooks are thin, one-execution-cell wrappers around the same tested script implementations. They do **not** define a second copy of the trainer. Each notebook contains a setting cell for selecting a condition/fold and running all or aggregating only.

## Interpretation and limitations

- **No paired-cell ground truth:** results measure population-level prediction fidelity.
- **Within-condition vs LOPO:** a drop in LOPO performance measures the out-of-condition transfer gap, but the training sets also differ in composition.
- **No-change baseline:** retaining NPC expression is a useful reference for MAE, but its predicted change vector is zero, so delta Pearson is undefined.
- **Top-k recovery:** ranking by `abs(mean(Neuron) − mean(NPC))` identifies strongly changing genes, **not** significance-filtered DEGs.
- **External validation:** cell-level held-out testing does not substitute for donor/replicate-held-out validation.
- **Feature-definition caveat:** LOPO uses a gene list defined from all conditions' headers. See `docs/METHODS.md` for details.
- The code has been syntax/structure checked but **not re-trained on public data in this environment**. You should verify your environment, data splits and results before publication.

## Citation and contribution

Please cite the Cell Translator paper when available and preserve attribution when reusing code. To provide contributions, open a pull request or issue on the corresponding repository after it is published. No public repository URL, authorship list, DOI, or license is asserted here.
