# Suggested work cards and source publication

These are proposed card contents, not updates to the live Trello board.

## Done: Verify the three-dimensional magnitude/sign representation

The completed preceding stage checked37 tests and six saved training records.
Keep its report separate; exact target roundtrips do not demonstrate prediction.

## Doing: Train conditional nonnegative transport with a grid sign predictor

Goal: learn the real error of the frozen calibrated dose predictor from CT and the
existing prediction. Begin with one record, then six records, then a train192 pilot.
Use weighted small-support transport banks, fixed full-grid source quadrature,
explicit positive-weight dynamics and separate sign supervision. Save complete
optimizer/RNG state and report finite-support approximation and compute cost.

Acceptance: training inputs exclude validation/test labels; model inference accepts
no target quantity; trajectories and losses are finite; train-only fitting is
meaningful; results and limitations are recorded. A weak positive improvement
boolean is not an overfitting certificate.

## To Do: Compare the new conditional model with the saved earlier systems

Use the same600 development records and uniform stored-array metrics for the new
model, previous Hahn-Jordan correction, ordinary signed RF, final Phase10D-strict,
and shared Phase9G. Retain raw predictions, case-level deltas and computation
metadata. Do not relabel the historical development set as a new independent test.
Report both overall errors and array-aligned dose-profile errors without hiding
regressions. The hard-sign row is a secondary sensitivity.

## To Do: Choose one evidence-based next modification

Use factor diagnostics to choose among conditioning/capacity, numerical integration,
coupling support, sign calibration or profile-aware training. Change one clearly
specified factor in a new experiment. More complex is not assumed to be better.

## Mac repository

The known Git repository is:
`/Users/ruicongrong/Documents/pr-hesser/ct2dose-project`
Do not use the manual same-name copy under master_thesis.

```bash
git -C "/Users/ruicongrong/Documents/pr-hesser/ct2dose-project" status -sb
```

After reviewing unrelated changes, create a new branch from the intended thesis
branch, or switch to it if already present (do not force):

```bash
git -C "/Users/ruicongrong/Documents/pr-hesser/ct2dose-project" switch -c thesis/conditional-wfr-dose
```

Extract the code ZIP under Downloads. Dry-run import first:

```bash
python3 "$HOME/Downloads/conditional_wfr_dose/scripts/install_into_repo.py" --repo "/Users/ruicongrong/Documents/pr-hesser/ct2dose-project"
```

Only when no conflicting files are reported, repeat with `--apply`. Stage only
`research/conditional_wfr_dose`, review the staged files and existing unpublished
history, commit, and push explicitly to the GitHub remote `github`, not GitLab's
`origin`. Never use a force push to solve a data/provenance problem.

```bash
git -C "/Users/ruicongrong/Documents/pr-hesser/ct2dose-project" add -- research/conditional_wfr_dose
git -C "/Users/ruicongrong/Documents/pr-hesser/ct2dose-project" --no-pager diff --cached --stat
git -C "/Users/ruicongrong/Documents/pr-hesser/ct2dose-project" diff --cached --check
git -C "/Users/ruicongrong/Documents/pr-hesser/ct2dose-project" commit -m "Add conditional WFR dose-correction pilot and paired evaluation"
git -C "/Users/ruicongrong/Documents/pr-hesser/ct2dose-project" push -u github thesis/conditional-wfr-dose
```

Do not publish medical arrays, record manifests, private paths, checkpoints,
optimizer state, source data or credentials. Check research sharing permissions
before publishing even aggregated figures. Code publication is not a backup of
private experimental results. Record the actual remote commit after push succeeds.
