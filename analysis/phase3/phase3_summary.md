# Phase 3 Summary

## 1. Background

The earlier 64/32 setup was used as a pilot experiment to verify that the 3D CT-to-dose pipeline works and that the task is learnable.

However, that setup was too small to support a reliable development-stage conclusion.

For this reason, the current phase moved to a larger train/validation setup while keeping the final test set fully held out.

---

## 2. Phase 3 split

### Case-level split
- 6 training cases
- 2 validation cases
- 2 hold-out test cases

### Sample-level split
- 2000 training samples
- 500 validation samples
- 4000 hold-out test samples saved for later final evaluation

The hold-out test set was not used in the current phase.

---

## 3. Baseline phase-3 result

The original larger-scale development run used:
- conditional 3D U-Net flow
- `base_ch = 24`
- `batch_size = 2`
- `lr = 3e-4`

This baseline already showed that:
- training on the 2000/500 setup is stable
- continuation beyond 30 epochs is beneficial
- the beam-shaped dose structure can be learned well on validation data

---

## 4. Controlled tuning

A first controlled tuning round was performed while keeping:
- the dataset split fixed
- the evaluation protocol fixed
- the model family fixed

The main hyperparameters explored were:
- learning rate
- base channel size

The strongest manually tuned configuration was:
- `lr = 5e-4`
- `base_ch = 24`
- `batch_size = 2`

This tuned run was then continued to 50 epochs.

---

## 5. Best current model result

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

## 6. Interpretation of training behavior

The larger-scale train/validation curves show that:
- training is stable overall
- no strong persistent overfitting signal is visible
- the 30-epoch setup had not yet saturated
- continuation to 50 epochs improves the result further
- controlled tuning improves the result beyond the original phase-3 baseline

Therefore, the model performance is not only sensitive to training duration, but also to the optimization setup.

---

## 7. Validation-wide error analysis for the best tuned model

Using the best tuned checkpoint, the validation-wide analysis gives:

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

## 8. Interpretation of validation behavior

The tuned model reconstructs the main beam-shaped dose structure very well overall.

In particular:
- the beam entrance region is localized correctly
- the along-beam decay pattern is learned very accurately
- typical cases now show relatively small percentage errors in both directions
- difficult cases are also improved clearly compared with the earlier untuned result

The main remaining limitation is most visible in harder perpendicular cross-sectional structure.

Therefore, the tuned result is stronger than the earlier untuned phase-3 result and should now be regarded as the strongest development-stage result obtained so far.

At the same time, the error is still not consistently below 1% for the full problem.

---

## 9. Wandb / Optuna-based tuning

A first wandb-based tuning workflow and a small Optuna study were completed successfully.

### Wandb result
The wandb runs confirmed that:
- `base_ch = 24` remains the strongest width
- the strongest learning-rate region is around `3e-4` to `5e-4`
- the worst candidate remains `1e-4`

### Optuna result
The Optuna study also pointed to the same strong region:
- best trial: `lr ≈ 4.2e-4`, `base_ch = 24`, `batch_size = 2`

The Optuna-selected configuration was then trained more fully to `50` epochs:
- best epoch: `40`
- best validation loss: `1.96e-05`
- final validation MSE: `2.40e-05`
- final validation MAE: `0.003286`

This Optuna-selected model did not outperform the current manually tuned best model.

Therefore:
- wandb / Optuna successfully confirmed the strong hyperparameter region
- but the strongest full training result still remains the manually tuned configuration:
  - `lr = 5e-4`
  - `base_ch = 24`
  - `batch_size = 2`
  - continuation to `50` epochs

---

## 10. Current conclusion

The current phase-3 result supports the following conclusion:

- the 3D flow model is stable on the larger 2000/500 development setup
- continuation beyond 30 epochs is clearly beneficial
- controlled tuning further improves the result
- wandb / Optuna confirm the same strong parameter region
- the best tuned checkpoint reconstructs the main beam-related dose structure very well
- both typical and difficult cases improve
- the main remaining limitation is hardest in the more difficult perpendicular cross-sectional structure

Thus, the current result should be understood as a strong development-stage result rather than only a pilot proof of concept.

It is still not the final formal evaluation, because the hold-out test set has not yet been used.

---

## 11. Suggested next discussion direction

The next project decision can be discussed along one of the following directions:
1. keep this as the current development-stage result and preserve the hold-out test set for a later formal stage
2. perform final hold-out evaluation in a later step
3. scale further if needed
4. prepare a later formal comparison against a diffusion-based baseline