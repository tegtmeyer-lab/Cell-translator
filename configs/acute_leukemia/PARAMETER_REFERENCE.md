# Acute leukemia parameter reference

The healthy differentiation (`CD34+ -> BMMC`) and leukemia transformation
(`CD34+ -> MPAL`) workflows use the same model and training settings. The two
workflows differ only in the target state and in which decoded output was used
for the final per-gene distribution analysis.

## Stage 1: independent-state HurdleAE

| Parameter | Value |
|---|---:|
| Latent dimension | 128 |
| Dropout | 0.10 |
| Batch size | 256 |
| Learning rate | 1e-3 |
| Maximum epochs | 200 |
| Early-stopping patience | 10 |
| Validation fraction | 0.20 |
| Seed | 42 |
| Nonzero input dropout | 0.15 |
| Detection threshold | 0.0 |
| BCE weight | 1.0 |
| Nonzero SmoothL1 weight | 1.0 |
| False-positive penalty weight | 2.0 |
| SmoothL1 beta | 0.1 |
| AdamW weight decay | 1e-4 |
| Gradient clipping | 1.0 |
| LR plateau factor / patience | 0.5 / 4 |

Each state is trained with its own HurdleAE. This is distinct from the jointly
trained shared HurdleAE used by several other Cell Translator applications.

## Stage 2: acute-leukemia CycleGAN variant

| Parameter | Value |
|---|---:|
| Batch size | 128 |
| Epochs | 300 |
| Gradient penalty | 10.0 |
| Critic updates | 3 |
| Drift epsilon | 1e-3 |
| Cycle weight | 10.0 |
| Identity weight | 0.1 |
| kNN pair weight | 5.0 |
| kNN k | 9 |
| Target bank sample | 5,000 |
| MMD weight | 2.0 |
| MMD sigmas | 1, 2, 4 |
| Covariance weight | 2.0 |
| InfoNCE weight | 0.0 |
| InfoNCE temperature | 0.07 |
| Supervised MNN weight | 0.0 |
| Learning rate | 1e-4 |
| LR exponential decay | 0.9 / 10,000 steps |
| Adam beta1 | 0.5 |
| Clipnorm | 1.0 |


