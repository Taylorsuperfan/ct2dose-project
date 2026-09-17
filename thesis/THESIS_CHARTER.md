# Master's Thesis Charter

## Official title

Conditional Flow Matching with Explicit Physical Transport Models for 3D CT-to-Dose Prediction

## Main objective

Develop explicit attenuation- and scattering-based physical dose models, and test whether conditional flow matching can refine their predictions into high-fidelity 3D dose distributions.

## Research questions

1. How accurately can explicit attenuation and scattering models approximate 3D dose from CT and beam information?
2. Does CFM refine a physics-based prior more effectively than direct residual regression?
3. Which explicit physical components matter most for depth-dose falloff and lateral-dose behavior?
4. Does the hybrid physics-CFM approach improve hard falloff cases under a strict case-level protocol?

## Minimum scope

1. Physical metadata and split audit
2. WED/attenuation physical baseline
3. Beam-aligned attenuation-diffusion baseline
4. Direct 3D regression baseline
5. Existing CT-direct CFM reproduction
6. Residual U-Net refinement from `D_phys`
7. Physical-prior CFM
8. Multiple seeds and case-level evaluation
9. Depth-dose, lateral-dose, and hard-falloff analysis

## Stretch goals

1. Physics-informed residual losses
2. Linear-vs-VP path ablation
3. Calibrated ray-based WED
4. Noisy stochastic interpolant
5. External test cases

## Out of scope unless explicitly required

1. Full Schrödinger Bridge
2. Unconstrained OT rematching of physical pairs
3. Full Boltzmann phase-space solver
4. Relativistic flow matching
5. Clinical-readiness claims
