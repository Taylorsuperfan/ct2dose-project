# Quick Links for April 7 Meeting

## Best 3D regression baseline

**Setup**
- 64 train / 32 test
- 3D U-Net
- `base_ch = 24`

**Metrics**
- Train MSE: `0.000250`
- Test MSE: `0.000319`
- Train MAE: `0.010088`
- Test MAE: `0.010948`

**Key files**
- Figure: `meeting/2026-04-07/regression_3d_best.png`
- Loss: `meeting/2026-04-07/regression_3d_loss.png`

**Interpretation**
- This is the current strongest baseline.
- It is the main reference model for comparison.

---

## Best 3D flow baseline

**Setup**
- 64 train / 32 test
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

**Key files**
- Figure: `meeting/2026-04-07/flow_3d_best.png`
- Loss: `meeting/2026-04-07/flow_3d_loss.png`

**Interpretation**
- This is the current best 3D flow baseline.
- It improves clearly over the earlier 3D flow versions.
- It still underperforms the current best 3D regression baseline.

---

## Euler-step check

**Steps tested**
- 30
- 50
- 100

**Result**
- The metrics and visual outputs remained almost unchanged.

**Conclusion**
- The current regression-vs-flow gap is not mainly caused by insufficient Euler sampling resolution.

**Key file**
- Figure: `meeting/2026-04-07/flow_euler_check.png`

---

## Monte Carlo status

- I have not run a separate Monte Carlo simulation myself.
- The current work focuses on learning from precomputed paired CT-dose cubes.
- Monte Carlo is currently better understood as a possible physics-based reference or validation step.

---

## Main interpretation

- The CT-to-dose mapping is learnable in both 2D and 3D.
- 3D regression is currently the stronger baseline.
- 3D flow is feasible and improves substantially with a more stable training setup.
- The remaining gap is more likely related to optimization stability than to Euler sampling resolution.

---

## Main discussion question

Should the project:
1. stop at the current best 3D flow baseline and document the comparison clearly, or
2. run one final controlled follow-up experiment under the same setup?

---

## Short oral version

I built and verified both 2D and 3D pipelines for the CT-to-dose task on paired `32×32×32` cubes. In 2D, regression performs better than flow. In 3D, I established a strong regression baseline and then built several conditional flow baselines under comparable settings. The current best flow model improved a lot with a more stable training setup, but it still remains below the regression baseline. I also checked whether the gap was mainly caused by Euler sampling, and that did not seem to be the main issue. So my current view is that the task is clearly learnable, regression is currently the stronger baseline, and flow is promising but still harder to optimize.