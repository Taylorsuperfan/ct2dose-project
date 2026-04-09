## Day 2 observations

### Regression visualization
- The regression prediction captures the main beam-shaped high-dose region and the overall dose decay pattern reasonably well.
- The predicted peak region is well aligned with the ground truth, although visible errors remain near the beam axis and around the dose boundary.
- Overall, the regression result is relatively sharp and close to the target structure.

### Flow visualization
- The flow prediction also captures the main beam-shaped dose region and the overall left-to-right decay trend.
- However, compared with the regression model, the flow result appears smoother and less sharp.
- The remaining errors are more visible near the high-dose region, along the beam axis, and around structural boundaries.

### Preliminary conclusion
- Both models learn the coarse dose structure.
- The regression model currently provides a sharper and more accurate reconstruction than the flow model.
- The flow model is clearly feasible, but the current output is still more diffuse and less precise in the peak-dose region.
- The strongest slice is consistently around slice 16 for the tested pilot samples, which makes it a useful reference slice for later beam-based analysis.