# Thesis Decision Log

## Project

**Title:** Conditional Flow Matching with Explicit Physical Transport Models for 3D CT-to-Dose Prediction

**Supervisor:** Prof. Dr. Jürgen Hesser

## Decisions

| ID | Date | Decision | Reason | Status |
|---|---|---|---|---|
| D001 | 2026-08-04 | This repository follows a thesis-only research scope. | Avoid mixing thesis work with unrelated projects. | Fixed |
| D002 | 2026-08-04 | Physical transport coordinate s is distinct from CFM model time t. | Preserve mathematical and physical interpretation. | Fixed |
| D003 | 2026-08-04 | Physical-model parameters may only be calibrated on train/validation data. | Prevent test-set leakage. | Fixed |
| D004 | 2026-08-04 | The final test set must remain untouched before model freeze. | Ensure independent final evaluation. | Fixed |
| D005 | 2026-08-04 | Physical baselines must be validated before increasing neural complexity. | Prioritize physical validity and thesis hypothesis testing. | Fixed |
