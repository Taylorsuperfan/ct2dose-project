# Phase 3 Summary

## 1. Background

The earlier 64/32 setup was used as a pilot experiment to verify that the 3D CT-to-dose pipeline works and that the task is learnable.

However, that setup was too small to support a more reliable development-stage conclusion.

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
- `30 epochs`

---

## 4. Main training result

### Best validation checkpoint
- best epoch: `26`
- best validation loss: `2.3445e-05`

### Final epoch
- final train loss: `5.2268e-05`
- final validation loss: `3.3849e-05`
- final validation MSE: `4.0283e-05`
- final validation MAE: `0.004768`

---

## 5. Interpretation of training behavior

The larger-scale train/validation curves show that:
- training is stable overall
- validation also improves strongly compared with the start of training
- no strong persistent overfitting signal is visible
- the best validation performance appears near the end of training rather than very early

This suggests that the current setup is meaningful beyond the earlier pilot phase.

---

## 6. Validation visualization

Using the best phase-3 checkpoint, representative validation examples show that:
- the beam-shaped dose structure is reconstructed well
- the beam entrance region is aligned correctly
- the along-beam decay pattern is reproduced accurately
- the perpendicular peak structure is also captured well

For one representative validation sample:
- reference slice: `16`
- beam center: `(peak_y = 16, peak_x = 0)`
- sample MSE: `4.0867e-05`
- sample MAE: `0.004205`

---

## 7. Current conclusion

The current phase-3 result supports the following conclusion:

- the current 3D flow model is stable on the larger 2000/500 development setup
- no strong overfitting signal is visible so far
- the best validation checkpoint reconstructs the main beam-related dose structure well
- both along-beam and perpendicular beam profiles are captured accurately on validation examples

Therefore, the current result is stronger than a pilot-scale feasibility result.

At the same time, it is still not the final formal evaluation, because the hold-out test set has not yet been used.

---

## 8. Suggested next direction

The next project decision can be discussed along one of the following directions:
1. keep this as the current development-stage result and prepare the next formal stage later
2. evaluate the final hold-out test set in a later step
3. prepare a later formal comparison against a diffusion-based baseline