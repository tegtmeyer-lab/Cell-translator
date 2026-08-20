# Leave-one-line-out morphology validation

This directory contains the validation-specific reduced model used for the control-cell leave-one-line-out (LOLO) analysis across five neural differentiation transitions.

## Transitions and selected held-out lines

The final five-line evaluation uses:

- Stem-->NPC: 10, 15, 20, 21, 9
- Stem-->Neuron: 19, 20, 21, 7, 9
- Stem-->Astrocyte: 10, 11, 12, 5, 6
- NPC-->Neuron: 12, 19, 20, 3, 5
- NPC-->Astrocyte: 10, 11, 12, 18, 5

Lines 22–25 are globally excluded before feature definition, model training, or held-out evaluation.

## Training configuration

- maximum 12,000 training cells per source/target domain
- maximum 5,000 held-out source and target cells
- AE: latent 128, dropout 0.03, 100 epochs, batch 128, learning rate 8e-5, patience 25, 15% internal validation
- AE objective: MSE + 0.05×mean-matching + 0.40×SD-matching; early stopping on validation MSE
- CycleGAN: 250 epochs, batch 64, 3 critic updates, GP=10, cycle=5, identity=0.5, moment=2
- separate source and target latent standardization
- all Python/NumPy/PyTorch/TensorFlow seeds set to 42 in each fold

Run selected folds from annotated morphology parquet files:

```bash
python -m analyses.leave_one_line_out.train \
  --config configs/validation/leave_one_line_out.yaml \
  --transition all
```

## Final five-line evaluation

The final post-hoc evaluator used archived fold matrices and reports effect-size Pearson correlation, median feature-wise KS, PCA variance log2 error, and discriminator AUC distance from 0.5:

```bash
python -m analyses.leave_one_line_out.evaluate_archived_folds \
  --config configs/validation/leave_one_line_out.yaml

python -m analyses.leave_one_line_out.plot \
  --results results/leave_one_line_out/archived_evaluation/lodo_five_line_corrected_metrics.csv \
  --outdir results/leave_one_line_out/figures
```
