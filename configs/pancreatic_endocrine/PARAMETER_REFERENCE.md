# Pancreatic endocrine application parameter reference

## Dataset preparation

- dataset: GSE132188 pancreatic development atlas
- source: `Ngn3 high EP`, day ≤ 14.5
- target: `Endocrine`, day ≥ 14.5
- source/target cell counts: 1,861 / 2,621
- atlas expression: already normalized/log1p in the source analysis; no additional normalization is applied
- HVG selection: Scanpy `highly_variable_genes`, `n_top_genes=5000`, `flavor="seurat"`, on the combined source/target subset
- marker force-inclusion uses mouse-case gene symbols from the notebook
- force-added genes in the executed run: `Gcg`, `Chgb`, `Kcnj11`
- final gene count: 5,003

The earlier 3,999-HVG atlas-mask extraction and an earlier uppercase marker block are exploratory notebook paths and are not used for the reported model.

## HurdleAE

- shared single-encoder HurdleAE
- latent dimension: 64
- dropout: 0.10
- batch size: 64
- learning rate: 5e-4
- maximum epochs: 300
- early-stopping patience: 25
- minimum improvement: 1e-4
- stratified validation fraction: 0.20, seed 42
- input nonzero-drop probability: 0.05
- BCE / magnitude / false-positive / soft-reconstruction weights: 1.0 / 5.0 / 2.0 / 0.5
- SmoothL1 beta: 0.1
- tail weight / clip: 1.0 / 6.0
- AdamW weight decay: 1e-4
- ReduceLROnPlateau factor / patience: 0.5 / 5
- gradient clipping: 1.0
- detection threshold: 1e-6
- positive-class weight clipped at 50
- masked SmoothL1 convention: `masked_mean + 1e-8` (`masked_smooth_l1_eps_mode: additive`)

## Latent CycleGAN

- batch size: 32
- epochs: 300
- WGAN-GP weight: 10
- critic drift: 1e-3
- critic updates per generator update: 5
- cycle / identity weights: 5.0 / 0.5
- kNN-pair / MMD / covariance / moment weights: 1.0 / 1.0 / 0.2 / 2.0
- kNN k: 5; target-bank sample: 8192
- MMD RBF bandwidths: 1, 2, 4
- moment loss uses the canonical symmetric standard-deviation formulation
- learning rate: 1e-4, exponential decay by 0.9 every 10,000 steps
- Adam beta1: 0.5; clipnorm: 1.0
- shuffle buffers: 4096 / 4096
- steps per epoch: target training cells // 32 = 65
- supervised MNN loss disabled (`lambda_sup=0`)
- TensorFlow was not explicitly seeded in the source notebook (`tf_seed: null`)
- reported prediction is generated from the held-out source subset (`translation.split: validation`)

## Evaluation

Gene-space KS/Wasserstein evaluation equalizes measured and translated target cell counts using `np.random.default_rng(42)`. The reported output used 372 translated source-validation cells and 525 measured target-validation cells.

## Fold-change analysis

The notebook contains two distinct definitions, retained separately:

1. Reported filtered summary: inverse-log1p linear means, expression threshold 0.1, detection fraction 0.05, pseudocount 0.1, log2FC clipped to ±6. This yielded `r=0.9875` and sign accuracy `0.8314`.
2. Later all-gene analysis: inverse-log1p linear means with pseudocount 1e-3 and a 1% detection mask used for visualization.

## Integrated Gradients / DriverScore

- IG steps: 64
- maximum source cells: 1,024
- batch size: 16
- baseline: source-cell mean
- seed: 42
- objective: `0.5 * ||G(z_std) - z_std||^2`
- DriverScore: absolute historical real log2FC
- DriverScore epsilon: 1e-8
- log2FC clip: ±6
- final detection filter: ≥10% in both source and measured target-validation cells
- category threshold: 75th percentile for DriverScore and normalized IG
