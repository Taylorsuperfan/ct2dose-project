# Long-Term Master's Thesis Context

## Official title

Conditional Flow Matching with Explicit Physical Transport Models for 3D CT-to-Dose Prediction

## Supervisor

Prof. Dr. Jürgen Hesser

## Advanced Practice starting point

The existing project contains:

- paired 3D CT-dose cube loading;
- conditional 3D U-Net;
- linear CT-to-dose flow matching;
- CT-initialized Euler sampling;
- global and profile-level evaluation;
- Phase9G multiplicative-additive calibration;
- Phase10D-strict falloff-aware refinement;
- a documented case-level split audit.

## Thesis main direction

1. Audit raw HU, physical dose units, spatial geometry, and beam metadata.
2. Implement explicit WED/attenuation and beam-aligned attenuation-diffusion models.
3. Use the physical model to generate a paired dose prior `D_phys`.
4. Train CFM to refine `D_phys -> D_target`.
5. Compare physics-only, direct regression, residual regression, CT-direct CFM, and physical-prior CFM.
6. Use strict case-level splits, multiple seeds, and common global/profile metrics.

## Methodological constraints

- Physical transport coordinate `s` is not CFM model time `t`.
- CFM velocity is not photon velocity.
- Preserve the true case/beam pairing between physical prior and target dose.
- Calibrate physical parameters using training/validation only.
- Do not inspect final test results during model development.
- Do not treat record count as independent patient count.
- Probability-path design is secondary unless the supervisor changes scope.
- Full OT rematching, Schrödinger Bridge, and full Boltzmann phase-space transport are not minimum scope.
- Do not invent missing physical metadata, units, or coefficients.
