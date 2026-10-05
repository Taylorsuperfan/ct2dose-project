# Source map and replay boundaries

## Current stage

`source/wfr_coefficient_refinement/` is the original delivered source distribution.
All Python modules, configuration and the unexecuted notebook are retained.
`refine/cache.py` constructs the frozen magnitude cache and train-only loss units.
`refine/losses.py` implements the A/B objectives; `selection.py` implements the
predeclared gate; `train.py` saves/resumes heads; `evaluation.py` locks choices and
predicts coefficients; `compare.py` produces paired saved-output comparisons.
Its `parent_source/cwfr/` modules match the original conditional parent byte-for-byte.

The R0--R10 notebook is under that package's `notebooks/`. It includes a checked
source payload, not checkpoint/data bytes. Open it for authorized reproduction,
not merely to prepare a Git upload. The original docs use future tense because
they are the pre-run plan; this release's stage report is the completed account.

## Conditional WFR parent

`source/conditional_wfr_dose/` contains the original N0--N9 pilot. It includes
weighted finite-support UOT coupling, path supervision, conditional motion/growth,
grid-sign inference and saved-baseline comparison. Its METHOD.md declares the
engineered source, boundaries, finite banks and nonzero-target scope.

## Earlier alternatives and Practical inference

`source/phase9g_signed_real_v1/pg9learn/` contains the actual direct residual and
Hahn--Jordan pilot implementations, not a newly invented proxy. `p10recover/`
contains the recovered Practical inference definitions and binding logic.
`records/source_provenance.json` records the original notebook segment identities.

The old Chinese instruction files/notebook and per-record historical numeric
table are intentionally not republished here. All old Python source modules are
retained unchanged. Some old archive-specific checks need the private historical
numeric table; do not interpret its omission as missing scientific implementation.
The current stage compares previously saved old predictions, so no baseline
retraining is required for the current result.

## Command-line examples for authorized environments

From `source/wfr_coefficient_refinement/`:

```bash
export PYTHONPATH="$PWD:$PWD/parent_source${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -v
python -m refine --help
```

From `source/conditional_wfr_dose/`:

```bash
python -m unittest discover -s tests -v
python -m cwfr --help
```

The original full experiment needs existing private cache/model artifacts.
Software tests instead create temporary synthetic fixtures and are not evidence
of real patient-level accuracy. Do not mix their numerical tables with `results/`.
