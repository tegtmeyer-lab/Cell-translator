# Cerebral organoid application parameter reference

The three organoid datasets use the same Cell Translator model/training implementation. Dataset-specific preprocessing and source/target maturation states are specified in separate YAML files.

## HurdleAE

- latent dimension: 128
- dropout: 0.10
- batch size: 128
- learning rate: 5e-4
- maximum epochs: 300
- early-stopping patience: 25
- minimum improvement: 1e-4
- validation fraction: 0.20, stratified, seed 42
- input nonzero-drop probability: 0.05
- BCE / magnitude / false-positive / soft-reconstruction weights: 1.0 / 5.0 / 2.0 / 0.5
- SmoothL1 beta: 0.1
- tail weight / clip: 1.0 / 6.0
- AdamW weight decay: 1e-4
- ReduceLROnPlateau factor / patience: 0.5 / 5
- gradient clipping: 1.0
- gene detection threshold: 1e-6
- per-gene positive-class weight clipped to [1, 50]

The source organoid notebook evaluates the masked SmoothL1 term as `masked_mean + 1e-8`; this convention is retained through `masked_smooth_l1_eps_mode: additive`.

## Latent CycleGAN

- batch size: 32
- epochs: 300
- WGAN-GP weight: 10
- critic drift: 1e-3
- critic updates per generator step: 5
- cycle / identity weights: 5.0 / 0.5
- kNN-pair / MMD / covariance / moment weights: 1.0 / 1.0 / 0.2 / 2.0
- kNN k: 5; target-bank sample: 8192
- MMD RBF bandwidths: 1, 2, 4
- learning rate: 1e-4 with exponential decay every 10,000 steps by 0.9
- Adam beta1: 0.5; clipnorm: 1.0
- steps per epoch: number of target-training cells divided by 32
- shuffle buffers: 4096 for source and target
- supervised MNN loss disabled (`lambda_sup = 0`)

The source notebook computes the standard-deviation part of moment loss as `abs(std_fake - std_real + 1e-6)`; this convention is retained through `moment_eps_mode: difference`.

TensorFlow was not explicitly seeded in the original notebook. Therefore retraining preserves the reported training procedure but cannot guarantee bitwise-identical CycleGAN weights. Exact numerical result reproduction should use archived trained artifacts when available.

## Dataset preparation

- Primary dataset: normalized 1–6 month matrices; 5,000 Seurat-v3 HVGs selected jointly with `batch_key=timepoint`; model pair 2 months → 6 months.
- HNOCA/He dataset: already log1p-normalized; exact ages 15, 30, 60, 90, 120, 150, 180 days; 5,000 Seurat-v3 HVGs; model pair 15 → 60 days.
- Independent dataset: already log1p-normalized; exact ages 4, 9, 12, 18, 21, 61 days; 5,000 Seurat-v3 HVGs; model pair 18 → 61 days.

No additional QC filtering or normalization was applied in the final preparation cells before HVG selection. Scanpy may warn that `seurat_v3` expects raw counts because the supplied matrices are normalized and the preprocessing choice is retained for fidelity.
