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

## 3. Model configuration

The current larger-scale development run uses:
- conditional 3D U-Net flow
- `base_ch = 24`
- `batch_size = 2`
- `lr = 3e-4`

The original phase-3 result was first trained for 30 epochs, and then continued in a controlled way to 50 epochs without changing the split or model definition.

---

## 4. Main training result after continuation to 50 epochs

### Best validation checkpoint
- best epoch: `49`
- best validation loss: `1.7157e-05`

### Final epoch
- final train loss: `3.7817e-05`
- final validation loss: `2.3082e-05`
- final validation MSE: `1.9592e-05`
- final validation MAE: `0.002913`

---

## 5. Interpretation of training behavior

The larger-scale train/validation curves show that:
- training is stable overall
- validation improves strongly compared with the beginning of training
- no strong persistent overfitting signal is visible
- the continuation from 30 to 50 epochs still gives meaningful improvement

This shows that the 30-epoch setup had not yet saturated, and that extending the training to 50 epochs is beneficial on the current 2000/500 development setup.

---

## 6. Additional validation error analysis

Following the professor’s suggestion, the validation result was further analyzed using:
- a loss curve rescaled to `0 ... 5e-4`
- profile plots with percentage error
- cross-sectional plots with matched color scales for ground truth and prediction
- a more sensitive difference scale
- explicit comparison between a typical validation example and a worst-case example

### Validation-wide summary
- the best validation samples are now around `3.46% – 3.61%` global relative error
- the overall validation picture is clearly improved relative to the earlier 30-epoch result

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

## 7. Interpretation of validation behavior

The current model reconstructs the main beam-shaped dose structure very well overall.

In particular:
- the along-beam decay pattern is learned very accurately
- the beam entrance region is localized correctly
- the perpendicular structure is also learned well for typical examples

The remaining difficulty is most visible in harder validation cases, especially in the perpendicular direction and in more difficult cross-sectional structure.

Therefore, the current result is stronger than a pilot-scale feasibility result.

At the same time, the errors are still not consistently below 1%.  
The current model performs very well overall, but the remaining errors are still in the few-percent range for typical cases and higher for difficult cases.

---

## 8. Current conclusion

The current phase-3 result supports the following conclusion:

- the current 3D flow model is stable on the larger 2000/500 development setup
- no strong persistent overfitting signal is visible so far
- training beyond 30 epochs is clearly beneficial
- the best validation checkpoint reconstructs the main beam-related dose structure well
- the along-beam behavior is learned especially well
- the main remaining limitation is hardest in the more difficult perpendicular cross-sectional structure

Therefore, the current result is now a strong development-stage result rather than only a pilot proof of concept.

It is still not the final formal evaluation, because the hold-out test set has not yet been used.

---

## 9. Suggested next discussion direction

The next project decision can be discussed along one of the following directions:
1. keep this as the current development-stage result and preserve the hold-out test set for a later formal stage
2. perform final hold-out evaluation in a later step
3. scale further if needed
4. prepare a later formal comparison against a diffusion-based baseline