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

## 2. Main question of the current stage

The goal of the current stage is to check whether the 3D flow model:
- trains stably on a larger setup
- still improves beyond 30 epochs
- reconstructs the beam-shaped dose structure well on validation data
- already shows acceptable error levels on typical and difficult cases

---

## 3. Phase-3 result after continuation to 50 epochs

### Configuration
- conditional 3D U-Net flow
- `base_ch = 24`
- `batch_size = 2`
- `lr = 3e-4`
- continued from 30 to 50 epochs

### Best result
- best epoch: `49`
- best validation loss: `1.7157e-05`

### Final epoch result
- final train loss: `3.7817e-05`
- final validation loss: `2.3082e-05`
- final validation MSE: `1.9592e-05`
- final validation MAE: `0.002913`

---

## 4. Main interpretation of the continuation result

The continuation from 30 to 50 epochs produced meaningful additional progress.

Compared with the earlier 30-epoch result:
- the best validation loss improved clearly
- the validation MSE improved strongly
- the validation MAE also improved substantially

This suggests that the model had not yet saturated at 30 epochs, and training beyond 30 epochs is still beneficial on the current 2000/500 setup.

---

## 5. Validation error analysis

### Validation-wide summary
- best validation samples are now around `3.46% – 3.61%` global relative error
- the current errors are therefore clearly improved compared with the earlier 30-epoch result

### Typical case
- sample index: `436`
- along-beam mean percentage error: `2.67%`
- perpendicular mean percentage error: `3.76%`

### Worst case
- sample index: `50`
- global relative error: `16.64%`
- along-beam mean percentage error: `3.90%`
- perpendicular mean percentage error: `12.13%`

---

## 6. Current conclusion

The current 3D flow model now shows a strong and stable development-stage result:

- it trains stably on the larger 2000/500 setup
- it still improves beyond 30 epochs
- the main beam-shaped dose structure is reconstructed very well
- along-beam behavior is learned particularly well
- the main remaining difficulty is in harder cross-sectional structure, especially in the perpendicular direction

At the same time, the error is still not consistently below 1%.  
So the current conclusion is:

the model learns the dose distribution very well overall, but the remaining errors are still in the few-percent range for typical cases and higher for difficult cases.

---

## 7. Current limitation

The hold-out test set has not yet been evaluated.

Therefore, the current conclusion should still be understood as a development-stage conclusion rather than the final formal evaluation.

---

## 8. Main discussion question

The main question for the next stage is what should be prioritized after this improved 50-epoch development result:

1. keep the hold-out test fully untouched for now,
2. move to final hold-out evaluation in a later formal stage,
3. scale further if needed,
4. or prepare a later formal comparison against a diffusion-based baseline.