# Mathematical Formulation

This document must be updated before each major model implementation.

## 1. Physical inputs

- CT representation and physical unit:
- Dose representation and physical unit:
- Voxel spacing/origin/orientation:
- Beam direction and source metadata:
- Beam entry definition:
- Beam energy/field information:

## 2. Physical transport coordinate

Let `s` denote physical beam depth and `r_perp` the transverse distance from the beam axis.

Define and verify:

- `s(x)`:
- `r_perp(x)`:
- coordinate transform from array indices to physical coordinates:

## 3. Physical model 1: WED/attenuation

- HU-to-density calibration source:
- WED definition:
- attenuation coefficient source/calibration:
- source profile:
- output quantity and unit:

## 4. Physical model 2: attenuation-diffusion

- state variable (fluence or dose surrogate):
- PDE:
- source condition:
- boundary conditions:
- numerical discretization:
- stability condition:
- parameter calibration protocol:

## 5. Conditional flow formulation

Recommended paired dose-field formulation:

- source endpoint: `x0 = D_phys`
- target endpoint: `x1 = D_target`
- condition: CT, beam metadata, and optional `D_phys`
- model time: `t in [0, 1]`
- path:
- target velocity:
- sampler:

## 6. Separation of concepts

- `s` is physical transport depth.
- `t` is CFM model time.
- Physical drift/fluence is not CFM tensor velocity.
- Physical parameter calibration must not use final test data.
