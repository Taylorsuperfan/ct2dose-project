# Quick Summary for April 16 Meeting

## Project
CT-to-Dose Prediction with 3D Conditional Flow on Paired 32×32×32 Cubes

---

## 1. Current stage

The earlier 64/32 setup should be understood as a pilot experiment only.

The current result is based on a larger development setup with:
- 6 training cases
- 2 validation cases
- 2 fully held-out test cases

At the sample level, the current development run uses:
- 2000 training samples
- 500 validation samples

The hold-out test set has not been used yet.

---

## 2. Main goal of the current stage

The goal of the current stage is to determine whether the 3D flow model:
- trains stably on a larger setup
- still improves beyond 30 epochs
- can be improved further through controlled hyperparameter tuning
- can be organized and searched more systematically with wandb / Optuna
- reconstructs the beam-shaped dose structure well on validation data
- achieves acceptable error levels on typical and difficult cases

---

## 3. Best current model

### Best tuned configuration
- `lr = 5e-4`
- `base_ch = 24`
- `batch_size = 2`
- continuation to `50` epochs

### Best result
- best epoch: `43`
- best validation loss: `1.5e-05`

### Final evaluation result
- final validation MSE: `1.7e-05`
- final validation MAE: `0.002662`

---

## 4. Main interpretation of training behavior

The larger 2000/500 setup is stable overall.

The continuation from 30 to 50 epochs gives meaningful additional improvement, which shows that the earlier 30-epoch setup had not yet saturated.

The first controlled tuning round also shows that the original configuration is not the best one. In particular, a slightly larger learning rate improves the result further.

---

## 5. Validation error analysis for the best tuned model

### Best case
- sample index: `246`
- global relative error: `2.91%`

### Typical case
- sample index: `264`
- global relative error: `5.48%`
- along-beam mean percentage error: `2.71%`
- perpendicular mean percentage error: `2.67%`

### Worst case
- sample index: `50`
- global relative error: `11.86%`
- along-beam mean percentage error: `3.06%`
- perpendicular mean percentage error: `7.45%`

### Averaged validation profiles
- average along-beam mean percentage error: `3.73%`
- average perpendicular mean percentage error: `1.13%`

---

## 6. Wandb / Optuna tuning result

A first tool-based tuning step was completed with both Weights & Biases and Optuna.

Both the wandb runs and the Optuna study pointed to the same strong hyperparameter region:
- `base_ch = 24`
- learning rate around `4e-4` to `5e-4`

The Optuna-selected configuration (`lr ≈ 4.2e-4`, `base_ch = 24`, `batch_size = 2`) was then trained more fully to `50` epochs. However, it did not outperform the current manually tuned best model.

So the current conclusion is:
- the tool-based search confirms the same strong parameter region
- but the best full training result still comes from the manually tuned configuration:
  - `lr = 5e-4`
  - `base_ch = 24`
  - `batch_size = 2`
  - continuation to `50` epochs

---

## 7. Current conclusion

The tuned 3D flow model now gives the strongest development-stage result obtained so far:

- training is stable on the larger 2000/500 setup
- continuation beyond 30 epochs is beneficial
- controlled tuning improves the result further
- wandb / Optuna confirm the same strong hyperparameter region
- the main beam-shaped dose structure is reconstructed very well
- both typical and difficult cases improve compared with the earlier result
- the main remaining difficulty is in harder perpendicular cross-sectional structure

At the same time, the error is still not consistently below 1% for the full problem.

So the current conclusion is:

the tuned model learns the dose distribution very well overall, but the remaining errors are still in the few-percent range for typical cases and higher for difficult cases.

---

## 8. Current limitation

The hold-out test set has not yet been evaluated.

Therefore, the current conclusion should still be understood as a development-stage conclusion rather than the final formal evaluation.

---

## 9. Main discussion question

The main question for the next stage is what should be prioritized after this tuned development-stage result:

1. keep the hold-out test fully untouched for now,
2. move to final hold-out evaluation in a later formal stage,
3. scale further if needed,
4. or prepare a later formal comparison against a diffusion-based baseline.