# Colab runbook

## Existing inputs (read only)

- Cache: `/content/drive/MyDrive/ct2dose_thesis/phase9g_signed_real_v1/cache.local/train192_seed42`
- Restored Practical evaluation: the `baseline_run` linked in that cache; expected
  `/content/drive/MyDrive/ct2dose_thesis/phase10d_recovered_v1/run01_val600`
- Prior HJD evaluation: `/content/drive/MyDrive/ct2dose_thesis/phase9g_signed_real_v1/evaluations.local/pilot_seed17_hjd_rf_eval01`
- Prior direct signed RF evaluation: `/content/drive/MyDrive/ct2dose_thesis/phase9g_signed_real_v1/evaluations.local/pilot_seed17_residual_rf_eval01`

No fallback silently regenerates old data, opens test cubes or guesses checkpoint
architectures. Missing old evaluation files must be restored from the actual saved
run before final comparison; they do not prevent training the new one-record stage.

## Independent notebook

Use `conditional_wfr_dose_colab.ipynb`, not a new cell hidden in the old WFR or
signed-grid notebook. The code is embedded and saved to a separate Drive code
backup. The new root is `/content/drive/MyDrive/ct2dose_thesis/conditional_wfr_dose`.

- **N0:** Restore and hash-check the English source.
- **N1:** Mount Drive, declare input/output locations and process environment.
- **N2:** Check/install only missing dependencies and execute CPU tests. An optional
  CPU comparison to installed POT is available. It is not a full author rerun.
- **N3:** Read the existing cache metadata and print the declared data selections.
  The cache constructor does not open any CT, dose or prediction arrays.
- **N4:** Set `RUN_ONE_RECORD=True` in that same cell. Prepare two small weighted
  training banks and run up to128 updates. Re-run the same cell for256 total.
  Review actual fitting, not just absence of exceptions.
- **N5:** Set `RUN_SIX_AFTER_REVIEW=True` only after reviewing N4. This is a fresh
  six-record run, not continuation from the one-record checkpoint. Three calls
  of128 updates complete384 updates. Both train and monitor are these six records.
- **N6:** Set `RUN_PILOT_AFTER_REVIEW=True` only after the one/six checks. Prepare
  two banks for each of192 training records. This is resumable after each bank.
  Start a fresh384-update model, monitoring40 validation records. The CPU bank
  solver does not require a GPU; training and repeated monitoring benefit from it.
- **N7:** After N6 reports completion, set `RUN_VALIDATION600=True`. This loads the
  selected saved weights, predicts600 records and saves raw/clipped/hard outputs.
  Previously completed per-record files are verified and reused.
- **N8:** Set `RUN_PAIRED_COMPARISON=True`. Read saved old predictions and create
  one joined comparison. No method is retrained in this step.
- **N9:** Display the new comparison and readable figures; copy no private record
  maps or medical arrays to a public repository.

A cell with a false switch prints NOT RUN; that is not an error. Change the switch
inside that original cell, otherwise a later `False` assignment will reset it.
Do not use Run all to launch all stages without reviewing the intermediate results.

## Expected output tree

```
conditional_wfr_dose/
  code/
  configs.local/
  logs.local/
  plans.local/{one,six,pilot}/
  runs.local/{one_seed17,six_seed17,pilot_seed17}/
  evaluations.local/pilot_seed17_val600/
  comparisons.local/pilot_seed17_vs_saved_baselines/
```

Coupling stats report approximate numerical convergence rather than assert exact
WFR optimality. Nonconverged iterate tests do not silently become `converged=True`.
The first small bank set is a declared approximation and can itself limit fitting.

## Review outputs

Training: `training_summary.json`, `monitor_history.csv`, `loss_history.csv`,
`best.json`, `last.json` and `COMPLETE.json`. A tiny positive RMSE change is NOT
sufficient evidence of overfitting. Review magnitude error, sign behavior and the
scale of the change on the one-record example.

Evaluation: `summary.csv`, `metrics_cases.local.csv`, `raw_metrics.local.csv`,
`factor_diagnostics.local.csv` and per-record prediction archives.

Comparison: `comparison.csv`, `paired_case_deltas.local.csv`, `compute_scope.csv`,
`report.md`, `figures/`. Metrics in the first new pilot have the same data but not
necessarily the same compute budget as old methods. No result is pre-filled.

## No retraining to solve a reconnect issue

If completed files exist, restore code/path variables and call only the unfinished
stage. Do not run old W6--W8 or the signed-grid bridge again. If runtime packages
are already available, dependency installation is not a prerequisite to reading
saved results. Explicit environment forks are described in RECOVERY.md.
