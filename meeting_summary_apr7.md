# Meeting Summary for April 7

## Project

CT-to-Dose Prediction: Regression vs Flow on Paired 2D/3D Cubes

---

## 1. Current goal

The goal of this project is to learn a mapping from paired CT cubes to dose cubes on the `32×32×32` dataset, and to compare regression-based and flow-based approaches under controlled 2D and 3D settings.

---

## 2. Data setup

I verified that the paired input/output cube structure is correct.

A patient-level 80/20 split was used:
- first 8 case folders for training
- last 2 case folders for testing

This avoids train/test leakage across patients.

---

## 3. Main progress so far

### 2D

- Built a 2D slice-based pipeline
- Verified that the CT-to-dose mapping is clearly learnable
- Found that the 2D regression baseline performs better than the 2D flow baseline

### 3D

- Built a 3D cube-based pipeline
- Established a strong 3D regression baseline
- Built multiple 3D conditional flow baselines under comparable settings
- Improved the 3D flow model substantially by using a more stable training setup

---

## 4. Current best 3D regression baseline

**Setup**
- 64 training samples / 32 test samples
- 3D U-Net
- `base_ch = 24`

**Metrics**
- Train MSE: `0.000250`
- Test MSE: `0.000319`
- Train MAE: `0.010088`
- Test MAE: `0.010948`

**Interpretation**  
This is currently the strongest baseline and serves as the main reference model.

---

## 5. Current best 3D flow baseline

**Setup**
- 64 training samples / 32 test samples
- conditional 3D U-Net flow
- `base_ch = 24`
- `batch_size = 2`
- `lr = 3e-4`
- `50 epochs`
- `30 Euler steps`

**Metrics**
- Train MSE: `0.000404`
- Test MSE: `0.000600`
- Train MAE: `0.013144`
- Test MAE: `0.015693`

**Interpretation**  
This is currently the best 3D flow baseline. It improves clearly over the earlier 3D flow versions, but it still underperforms the current 3D regression baseline.

---

## 6. Additional checks

### Euler step check

I re-evaluated the same 3D flow checkpoint with different Euler step counts:
- 30
- 50
- 100

### Result

The metrics and visual results remained almost unchanged.

### Conclusion

This suggests that the current gap between 3D flow and 3D regression is not mainly caused by insufficient Euler sampling resolution.

---

## 7. Current interpretation

At the current stage:
- the CT-to-dose mapping is learnable in both 2D and 3D
- 3D regression is still the stronger baseline
- 3D flow is clearly feasible
- 3D flow improves substantially with a more stable training setup
- the remaining gap is more likely related to optimization stability than to Euler sampling resolution

---

**In short:** regression is currently the stronger baseline, while flow is promising but still harder to optimize.

---
## 8. Monte Carlo status

At the moment, I have not run a separate Monte Carlo simulation myself.

The current experiments focus on learning from precomputed paired CT-dose cubes and comparing regression-based and flow-based models.

My current understanding is that Monte Carlo would be more relevant as a physics-based reference or validation step for the present prediction pipeline.

---

## 9. Main discussion question

At this stage, I would like to discuss whether it would be more meaningful to:
1. stop at the current best 3D flow baseline and document the comparison clearly, or
2. run one final controlled follow-up experiment under the same setup.

---

## 10. Short oral version

I verified the paired cube data structure and built both 2D and 3D pipelines for the CT-to-dose task. In 2D, regression clearly outperforms flow. In 3D, I established a strong regression baseline and then built several conditional flow baselines under comparable settings. The current best flow model improves substantially with a more stable training setup, but it still underperforms the regression baseline. I also checked whether the remaining gap was mainly due to Euler sampling, but increasing the number of Euler steps did not help much. So my current interpretation is that optimization stability is more important than sampling resolution. I have not run a separate Monte Carlo simulation yet; so far, the work focuses on learning from precomputed paired CT-dose data. I would like to discuss whether the current comparison is already sufficient for this stage, or whether one final controlled follow-up experiment would still be worthwhile.