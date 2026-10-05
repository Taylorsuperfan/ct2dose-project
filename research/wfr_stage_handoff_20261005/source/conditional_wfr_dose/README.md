# Conditional 3-D WFR magnitude transport with a grid-valued sign predictor

A research adapter for the **existing real Phase9G-error task**, with a complete
training, checkpoint/recovery, validation and saved-baseline comparison path.
It is not a guaranteed improvement, a water-reference experiment, an independent
final test, or a clinical dose engine.

Start with `notebooks/conditional_wfr_dose_colab.ipynb`. Run one training record,
then six records, then the declared train192/monitor40 pilot. Each stage starts
from fresh parameters, unless an explicit continuation is requested. Old WFR
simulation and Practical files are never overwritten or retrained.

## Included

- Lazy, hash-checked read access to the existing train192/validation600 cache.
- Finite-support **weighted** OET banks and analytic travelling-Dirac paths.
- CT/base-conditioned velocity and growth networks, plus an independently
  parameterized, supervised 3-D sign network.
- Deterministic 32^3 source quadrature, nonnegative log-mass evolution, and
  trilinear magnitude deposition followed by grid-sign multiplication.
- Target-free inference: only CT and the frozen prediction enter `predict`.
- Checkpoints with optimizer, RNG and monitor history; explicit cross-environment
  continuation records instead of contract edits.
- Paired comparison to saved Phase10D-strict, saved Hahn-Jordan correction,
  saved direct signed RF, and their shared Phase9G starting prediction.
- Raw-dose and clipped-dose metrics, global and profile metrics, factor
  diagnostics, compute budgets and descriptive plots.

## Read before interpreting results

`docs/METHOD.md`, `docs/COMPARISON_PROTOCOL.md`, `docs/SEVEN_DAY_PLAN.md`,
`docs/RECOVERY.md`, and `verification/VERIFICATION.md` distinguish source-derived
mathematics, proposed engineering choices, local tests and unexecuted experiments.

This is a new **conditional adapter**, not unchanged author WFR-FM code. Training
uses bounded approximate couplings, not full 32768-by-32768 OT. Exact all-zero
residual records are deliberately rejected by this first nonzero-target pilot.
Zero-target representation success does not solve zero-mass WFR dynamics.

## Requirements

Python 3.10+, NumPy, PyTorch, pandas, matplotlib, tabulate. SciPy is used by one
independent numerical test. POT is optional for the small external reference
check; it is not used by the trainer and cannot preallocate this trainer's GPU.
Do not upgrade a working PyTorch installation merely to run this package.

All output arrays, record IDs, checkpoints and data manifests remain private.
