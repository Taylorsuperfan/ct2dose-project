# Seven-day execution plan

The plan is conditional on passing each stage's scientific review, not a promise
that training or all goals will finish in seven days. Do not redo the completed
2-D author experiment or 3-D codec stage.

| Day | Work | Notebook | Evidence / stop condition |
|---|---|---|---|
|1|Restore new code, run CPU tests and optional installed-POT reference; read method/protocol; prepare one-record banks.|N0--N4|No medical validation arrays opened in preparation; record approximate coupling convergence.|
|2|Train on one permitted real training record in two128-update calls (256 total). Read train-only magnitude/sign/dose behavior.|N4|Finite gradients and trajectories, known source total, actual nontrivial fitting rather than a barely positive boolean. If it fails, debug this stage only.|
|3|Train a fresh model on six fixed train records, one per case,384 total updates. No validation arrays.|N5|Check different conditions and all six cases; no claim of generalization.|
|4|Prepare train192 weighted banks, review approximate support/solver diagnostics; start fresh384-update pilot.|N6|Use the unchanged existing scale. No separate target-total normalization.|
|5|Complete pilot, inspect40-record monitor history, then run saved-checkpoint inference on validation600.|N6--N7|Do not stop early because a favorable checkpoint appeared; complete the declared budget.|
|6|Run paired comparison against saved HJD, ordinary signed RF, Phase10D-strict and shared predictor.|N8|Check identical IDs and units, paired case deltas, raw/clipped and profile metrics. No old training.|
|7|Write findings, costs, limitations and one next modification supported by diagnostics.|N9 and docs|Freeze original outputs, record a decision; do not adjust many components together.|

## If the model does not improve

- Bad fit on one record: inspect conditioning, coupling approximation, growth and
  sign errors before increasing the full training size. A boolean `learned_beyond_zero`
  can be true for a negligible change; it is not an overfitting criterion.
- One record fits but six do not: investigate conditioning/capacity and contradictory
  local transport supervision. Preserve the failed run; change one declared factor.
- Magnitude good but signed residual poor: inspect class recalls, near-zero regions
  and soft-sign shrinkage. Do not inject the true sign or true total into inference.
- Global errors improve but profiles worsen: propose a profile-aware NEXT experiment
  with explicit model-selection rules. ODE endpoint backpropagation, if introduced,
  changes the claim of purely simulation-free training and must be labeled.
- Results may depend on16 integration steps: perform a fixed-checkpoint8/16/32-step
  convergence study on a fixed training/development subset in a separate evaluation;
  do not choose the best600-row outcome after the fact.
- More coupling supports/banks may help representation but cost more. Preserve common
  endpoint totals and report the altered approximation; copying identical particles
  does not add spatial resolution.
- A1536-update extension can be an explicitly labeled learning-curve experiment,
  not the same-budget comparison. Use the fork tool and retain cumulative updates.

Only after the method is understood on this established real residual task should
you scale data, add a genuine water-reference field, or arrange a truly unseen test.
