# WFR coefficient refinement: completed-stage handoff

This release records one completed development experiment, not a final clinical model.
The study keeps a conditional Wasserstein--Fisher--Rao (WFR) magnitude predictor
fixed and asks whether a better supervised bounded signed coefficient improves
the actual dose correction. Two paired arms were run: composition supervision
alone (A), and composition plus array-x profile supervision (B).

The main finding is a profile improvement with a small whole-cube trade-off.
B reduces x-line mean percentage error from 8.3497% for the parent to 7.7271%,
while whole-cube RMSE changes from 4.04396e-6 to 4.04510e-6 stored units.
Relative to A, B reduces x-line percentage error by 3.055%, with a 0.094%
whole-cube RMSE increase. It still does not surpass the saved final Practical
system or the separate positive/negative model on the principal x-profile measure.

These 600 cubes come from two development cases and include the 40 selection
records. They are not 600 independent patients. Physical dose units, beam axes,
water-reference performance and clinical readiness have not been established.

## Read first

- [Stage report](docs/STAGE_REPORT.md): motivation, implementation, results and interpretation.
- [Method comparison](docs/METHOD_COMPARISON.md): WFR, Hahn--Jordan, direct residual flow and Practical baseline.
- [Next work](docs/NEXT_STEPS.md): planned repeats and bounded architectural hypotheses.
- [GitHub steps](docs/GITHUB_UPLOAD.md): exact local workflow without replacing the repository.
- [Source map](docs/SOURCE_MAP.md): where each implementation and notebook lives.
- [Private prerequisites](docs/PRIVATE_ARTIFACTS.md): what remains outside this public-oriented package.
- [Verification](verification/RELEASE_VERIFICATION.md): checks actually run during packaging.

## What is included

`source/wfr_coefficient_refinement/` contains the original refinement training,
evaluation, checkpoint-selection and comparison code. Its parent `cwfr` modules
are retained byte-for-byte. `source/conditional_wfr_dose/` contains the original
conditional 3-D pilot, including its numerical UOT coupling implementation.
`source/phase9g_signed_real_v1/` contains the previous direct residual and
positive/negative methods plus the recovered Practical inference modules.
All copied Python modules are unchanged. See `provenance/source_archives.json`.

`results/` contains aggregate numeric tables transcribed from the user's supplied
stage report, at the precision displayed in that report. This packaging process
did not recompute medical predictions. `figures/` contains the three supplied
aggregate plots unchanged. Do not mistake source tests with synthetic fixtures
for these reported medical-development measurements.

## Check the handoff

```bash
python3 scripts/check_release.py
```

This is a standard-library check. It does not train a model, open private arrays,
install packages, contact a service or publish anything.

The training entry point is
`source/wfr_coefficient_refinement/notebooks/wfr_coefficient_refinement_colab.ipynb`.
Do not rerun completed training merely to upload this release. An authorized
research environment with the existing caches and checkpoints is needed to
reproduce the recorded study. Missing artifacts are not replaced with synthetic
predictions.

## Publication boundary

No CT/dose arrays, checkpoints, patient/case identifiers, per-record metrics,
private transcript/PDF, or executed notebook outputs are included. This is not
a guarantee of permission to publish aggregate research results or inherited
repository history. Obtain the appropriate approval and review the exact staged
files before pushing. No project-wide license is assigned by this release.
