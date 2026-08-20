# Cell Painting morphology parameter reference

This file records the executable settings used in the coherent 3,409-feature
Cell Painting workflow.

## Morphology autoencoder

| Setting | Value |
|---|---:|
| Latent dimension | 256 |
| Epochs | 100 |
| Batch size | 128 |
| Learning rate | 1e-4 |
| Validation fraction | 0.20 |
| Split random state | 42 |
| Early-stopping patience | 20 |
| Adam weight decay | 1e-5 |
| LR scheduler factor | 0.5 |
| LR scheduler patience | 8 |
| Gradient clip | 1.0 |
| Training variance-loss weight | 2.0 |
| Training covariance-loss weight | 0.1 |
| Validation objective | MSE only |

The model is state-specific: Stem, NPC, Neuron, and Astrocyte each receive an
independently trained AE with the same architecture and hyperparameters.


## Morphology CycleGAN

| Setting | Value |
|---|---:|
| Epochs | 300 |
| Batch size | 32 |
| Gradient-penalty weight | 10.0 |
| Critic updates | 5 |
| Drift epsilon | 1e-3 |
| Cycle weight | 5.0 |
| Identity weight | 0.5 |
| kNN-pair weight | 1.0 |
| MMD weight | 1.0 |
| Covariance weight | 0.2 |
| Moment weight | 2.0 |
| kNN k | 5 |
| Target-bank sample | 8192 |
| MMD sigmas | 0.5, 1.0, 2.0 |
| Learning rate | 1e-4 |
| LR decay steps | 8000 |
| LR decay rate | 0.9 |
| Adam beta1 | 0.5 |
| Clipnorm | 1.0 |

Source and target latent spaces are standardized **separately**, using the
training mean and standard deviation of each state. The morphology generator
is 256->512->256->256->256 with LayerNormalization and ReLU; the forward model
has an additional affine output head. The critic is 256->256->128->64->1.


## Interpretation

Integrated Gradients uses 64 integration steps, at most 1,024 source cells, a
mean baseline, batch size 16, and seed 42. DriverScore is the absolute measured
Cohen's d. 
