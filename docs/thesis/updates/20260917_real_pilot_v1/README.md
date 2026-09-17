# Real Phase9G residual pilot V1 — result update (2026-09-17)

**Experiment run: 2026-09-16. Status: completed development comparison, not demonstrated superiority on the main profile objective.**

This update records the user-reported results of the completed real-data pilot. It does not change either released source package, its original hashes or its status-at-delivery documents.

- Recovered baseline: Phase5E RF -> Phase9D-plus -> Phase9G calibration -> Phase10D-strict.
- New target: real GT minus **frozen Phase9G prediction**. Phase9G is NOT water dose.
- New corrections: ordinary signed residual RF and spatial HJD RF (two flows + mass heads + overlap/endpoint objectives).
- Training: 192 records across six training cases; 384 optimizer updates per new method; batch size two; seed 17. A separate 12-record learnability check did not provide warm-start weights to the pilot.
- Selection: best equal-case full-volume RMSE on 40 validation records.
- Reporting: 600 records from the same two validation cases, including the selection subset. Development evidence only; not 600 patients and not a blind final test.

## Result

Compared with Phase10D-strict:

| New method | Full-volume RMSE change | Full-volume MAE change | x profile RMSE change | x mean percentage-error change |
|---|---:|---:|---:|---:|
| Ordinary residual RF | -4.20% | -7.43% | +3.45% | +1.220834 percentage points (+18.29% relative) |
| HJD RF | -3.47% | -4.99% | +2.34% | +0.594450 percentage points (+8.91% relative) |

Both new methods have slightly higher aggregate x/y/z absolute profile RMSE than Phase10D-strict. HJD improves x-profile errors compared with ordinary residual RF, but does not improve x-profile errors compared with Phase10D-strict or frozen Phase9G.

**Conclusion: useful global correction, not a complete improvement of the original profile objective. The x regression is not solely an artifact of percentage-error denominators.** This does not locate the failing region or establish a sole cause.

The old Phase9D-plus component has the lowest absolute x/y/z profile RMSE in the supplied table; Phase10D-strict has the lowest x mean percentage error. Retain all components rather than changing the baseline after observing results.

## Boundaries

Legacy axes remain x=W, y=H, z=D for GT-peak diagnostic lines, not verified beam/mm geometry. Old feature/training x=D behavior remains disclosed. Absolute line RMSE uses the whole line; percentage error masks positions below 1% of the target-line peak. Dose units are stored numerical units, not a certified Gy scale. Main reported dose is clipped nonnegative; raw outputs remain private diagnostic evidence.

One training seed and two validation cases do not establish significance or clinical generalization. Historical upstream data exposure remains unresolved; strict loading does not certify blind end-to-end independence. New and old methods have different historical training budgets. Current CUDA warnings preclude a bitwise-determinism claim.

## Files and priority

`comparison_reported.csv` and `relative_changes_vs_phase10d.csv` contain aggregate-only result transcriptions/calculations. `experiment_registry.csv`, `decision_log.md` and `weekly_plan.md` give the current state. The older package `records/experiment_registry.csv` files are immutable **status-at-delivery** records, not the latest state; do not edit them simply to replace pending with done.

No V2 training was launched. Proposed next work: predefine profile-aware final-dose losses and checkpoint-selection constraints for both new methods, then run a separately versioned experiment without overwriting V1.
