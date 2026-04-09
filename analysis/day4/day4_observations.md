## Day 4 observations

### Signed velocity maps
- The signed velocity maps are clearly structured rather than random.
- The strongest positive updates are concentrated near the beam entrance and the peak-dose region.
- Away from the main beam structure, the update magnitude becomes much weaker.
- Small negative updates appear in downstream and peripheral regions, suggesting local correction rather than uniform amplification.

### State evolution
- The evolving state starts from the CT-conditioned initialization and gradually develops the beam-shaped dose structure over time.
- The final state at later time points is consistent with the previously observed flow prediction.

### Along-beam velocity profile
- The largest positive updates occur near the beam entrance.
- The update magnitude decays rapidly downstream.
- In later downstream regions, the velocity becomes slightly negative, which suggests that the model applies weak corrective suppression there.

### Perpendicular velocity profile
- The strongest positive updates are sharply concentrated near the beam center.
- Away from the beam center, the update magnitude decreases quickly and becomes slightly negative in peripheral regions.
- This indicates that the learned update pattern is spatially localized around the main high-dose structure.

### Preliminary conclusion
- The current flow behavior is not random: the model applies structured voxel-wise updates near the beam entrance, beam axis, and peak-dose region.
- This supports the interpretation that the flow model has learned a meaningful update mechanism.
- At the same time, the final flow output remains smoother than the regression baseline, which suggests that the remaining limitation is not the absence of structure, but insufficient sharpness and peak preservation in the final reconstruction.