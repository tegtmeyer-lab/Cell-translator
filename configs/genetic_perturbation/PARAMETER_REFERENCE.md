# Genetic perturbation parameter reference

This file records the parameters used by the final genetic-perturbation workflow
and provides a human-readable companion to `common.yaml`. Values were checked
across the five control→perturbation blocks in the source analysis.

## Stage 1: separate-encoder HurdleAE

The five perturbations use the same Stage-1 configuration.

| Setting | Value |
|---|---:|
| Random seed (NumPy/PyTorch) | 42 |
| Latent dimension | 128 |
| Encoder dropout | 0.10 |
| Batch size | 64 |
| Learning rate | 5e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Maximum epochs | 300 |
| Early-stopping patience | 25 |
| Early-stopping min delta | 1e-4 |
| Detection threshold | 1e-6 |
| Nonzero input-drop probability | 0.05 |
| Gradient clipping | 1.0 |
| Validation fraction | 0.20 per domain |
| Train loader `drop_last` | True |
| Per-gene `pos_weight` maximum | 50 |
| Inference batch size | 256 |
| LR plateau factor | 0.5 |
| LR plateau patience | 5 |

Hurdle loss weights:

| Term | Weight |
|---|---:|
| detection BCE | 1.0 |
| nonzero magnitude | 5.0 |
| false-positive penalty | 2.0 |
| soft reconstruction | 0.5 |
| SmoothL1 beta | 0.1 |
| tail upweighting alpha | 1.0 |
| tail clip | 6.0 |

Architecture:

```text
control encoder: input → 512 → 256 → 128 → latent(128)
target encoder:  input → 512 → 256 → 128 → latent(128)
shared decoder:  latent → 128 → 256 → 512 → {detection logits, magnitude}
```

Encoder hidden blocks use Linear → LayerNorm → GELU, with dropout 0.10 in the
512 and 256 blocks. Decoder hidden blocks use Linear → LayerNorm → GELU. The
magnitude output uses Softplus and the detection output uses Sigmoid.

**Training-loop behavior retained for fidelity:** source and target DataLoaders
are consumed with `zip()`. Both train loaders use `drop_last=True`, so the
smaller domain determines Stage-1 updates per epoch.

## Stage 2: latent WGAN-GP CycleGAN

The five perturbations use the same Stage-2 configuration.

| Setting | Value |
|---|---:|
| Batch size | 32 |
| Epochs | 300 |
| Critic updates per generator update | 5 |
| Gradient penalty | 10.0 |
| Critic drift epsilon | 1e-3 |
| Cycle weight | 5.0 |
| Identity weight | 0.5 |
| k-NN pair weight | 1.0 |
| MMD weight | 1.0 |
| Covariance weight | 0.2 |
| Moment weight | 2.0 |
| Supervised MNN weight | 0.0 (inactive) |
| k for pair loss | 5 |
| Target-bank sample | 512 |
| RBF-MMD sigmas | 4, 8, 16, 32 |
| Initial LR | 1e-4 |
| LR decay steps | 10,000 |
| LR decay rate | 0.9 |
| Adam beta1 | 0.5 |
| Gradient clip norm | 1.0 |
| Source shuffle buffer | 4,096 |
| Target shuffle buffer | 512 |
| Steps per epoch | `len(source_train) // 32` |
| MNN refresh interval | 10 epochs |
| MNN pool | 400 |
| MNN target sampling replacement | True |
| TensorFlow random seed | not set in source analysis |

The forward generator has Dense(1024) → Dense(512) → Dense(512) → Dense(latent)
with ReLU + LayerNorm on the hidden layers, followed by a bare affine output
head. The backward generator uses the same core MLP without the additional
forward affine head. Critics use Dense(512) → 256 → 128 → 1 with ReLU hidden
activations and no output sigmoid.

Because the original TensorFlow RNG was not seeded, archived Keras models are
the path for exact numerical reproduction of the reported predictions. Fresh
retraining preserves the procedure and hyperparameters but is stochastic.

## Evaluation definitions

The notebook contains multiple effect summaries used for different figures.
They are retained under explicit names rather than merged into one definition.

1. **Single-perturbation historical LFC:**
   `log1p(mean(target_log1p)) - log1p(mean(control_log1p))`.
2. **Final dedicated log2FC analysis:** inverse-log1p to linear expression,
   average per gene, then `log2((target_mean + 1e-3)/(control_mean + 1e-3))`.
   The final visualization uses a 0.01 detection filter for its top panels.
3. **Final cross-model effect recovery:** mean log1p perturbation minus mean
   log1p control; Pearson correlation is evaluated over the top 5,000 genes by
   absolute measured effect.

An earlier exploratory log2FC block used pseudocount 0.1 together with
expression/detection filtering. It is superseded by the later repeated
linear-space analysis above and is not used as the canonical reproduction path.

## Integrated gradients / DriverScore

| Setting | Value |
|---|---:|
| IG integration steps | 64 |
| Maximum source cells | 1,024 |
| Baseline | mean source cell |
| IG batch size | 16 |
| IG sampling seed | 42 |
| DriverScore `|real effect|` weight | 1.00 |
| DriverScore `|pred effect|` weight | 0.40 |
| Good-match weight | 0.75 |
| Sign-match weight | 0.35 |
| Shift-error weight | -0.40 |
| Good-match absolute predicted-effect threshold | 0.10 |
| DriverScore epsilon | 1e-8 |
| Minimum detection | 0.10 |
| Effect clip | ±6 |
| Driver/IG quantile threshold | 0.75 |
| Top genes | 40 |
| Labels | 20 |

## Fixed numerical constants

These values are implementation constants in the original formulas rather than
tunable application hyperparameters. They are recorded in `common.yaml` under
`implementation_constants` so they are not hidden from the reproducibility
record.

| Constant | Value |
|---|---:|
| Latent standardization epsilon | 1e-6 |
| Gradient-penalty norm epsilon | 1e-12 |
| Moment-loss epsilon | 1e-6 |
| Driver IG min-max epsilon | 1e-12 |
| Driver z-score epsilon | 1e-12 |
| Latent-alignment variance denominator epsilon | 1e-8 |

A stale comment in the Stage-2 notebook described some saved latents as 64-D.
The executable Stage-1 configuration uses `LATENT_DIM = 128`, and the captured
Stage-1 output reports `Z_ctrl shape: (10529, 128)`. The reproducibility package
therefore records the actual latent dimension as **128**.

## Software-version provenance

The source notebook identifies a Python 3.11 user environment in its captured
runtime output, but it does not record exact NumPy, PyTorch, TensorFlow,
scikit-learn, SciPy, Scanpy, or related package versions. Exact dependency
versions should therefore only be pinned if the original environment metadata
can be recovered; version numbers should not be guessed.
