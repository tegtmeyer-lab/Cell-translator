# Methods and interpretation

## Analysis1: Within-condition developmental translation

Six independent NPC-> Neuron models are fit, one each for control, APOE, MSI2, NR1H3, ANKRD12 and LRRC4C. For each state/condition, cells are split 70/15/15 for training, validation and test. Train-only AE thresholds and latent standardization are used. A condition-specific HurdleAE and latent CycleGAN learn the state translation. Final metrics are computed on held-out test populations.

## Analysis2: Leave-one-perturbation-out (LOPO)

For each of five perturbations, independently fit a pooled developmental model using control plus four other perturbations. Neither NPC nor Neuron examples from the withheld condition are used for fitting or validation. Held-out NPC test cells, frozen in analysis1, are translated after training; held-out Neuron cells are used only for evaluation.
