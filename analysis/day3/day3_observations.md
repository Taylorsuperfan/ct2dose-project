## Day 3 observations

### Cross-sectional visualization
- Both models capture the main beam-shaped dose structure on the reference slice.
- The regression result remains sharper and closer to the ground truth.
- The flow result is smoother and more diffuse, especially around the beam axis and boundary regions.

### Along-beam profile
- The regression profile follows the main peak and the downstream decay pattern very closely, although it slightly overestimates the dose near the beam entrance.
- The flow profile also captures the overall decay trend, but it underestimates the peak near the beam entrance more clearly than the regression model.

### Perpendicular profile
- The regression profile preserves the lateral structure well and matches the peak position accurately, although the peak is slightly overestimated.
- The flow profile also captures the overall lateral shape, but the peak is underestimated and appears more smoothed.

### Preliminary conclusion
- Beam-based visualization confirms that both models learn the main beam-related dose structure.
- The regression model is currently more accurate in the peak-dose region and preserves sharper spatial structure.
- The flow model is clearly feasible and captures the correct global trend, but it still appears more diffuse and tends to underestimate the peak intensity.
- The reference beam center at `slice = 16`, `peak_y = 16`, and `peak_x = 0` is physically consistent with a beam entering from the left side of the image.