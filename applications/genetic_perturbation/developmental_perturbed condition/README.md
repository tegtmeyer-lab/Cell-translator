# Cell Translator — Developmental Translation and Unseen-Perturbation Generalization

Research implementation of two complementary experiments using a Hurdle Autoencoder (HurdleAE) and latent-space CycleGAN. 

## Research questions

**Analysis 1 - Developmental prediction within known conditions:** independently learn NPC --> Neuron translation for control, APOE, MSI2, NR1H3, ANKRD12 and LRRC4C. This is a within-condition held-out-cell evaluation; the target state is represented in model training.

**Analysis 2 — Leave-one-perturbation-out (LOPO):** train a new pooled NPC --> Neuron model on control and four perturbed conditions, excluding **both NPC and Neuron cells** of the fifth perturbation. Translate that perturbation's frozen NPC test population and compare against its frozen measured Neuron test population. Repeat for all five holdouts. This evaluates cross-perturbation transfer.

## Installation

Requires a Python environment with compatible **PyTorch and TensorFlow/Keras** installations. Dependencies are listed in `requirements.txt`; GPU-specific installation of PyTorch/TensorFlow may need adjustment for your compute environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/config.example.json configs/config.json
```

Edit `configs/config.json` to specify input CSVs and output directory. 

Each CSV contains cells as rows, genes as named columns, and an index/barcode in the first column. 

## Citation and contribution

> **Citation:** If you use or adapt this code or approach, please cite the our Cell Translator manuscript.
