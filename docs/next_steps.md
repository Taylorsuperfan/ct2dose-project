# Next Steps

This document lists the recommended next steps after the current Phase10D-strict split-corrected diagnostic result.

---

## 1. Evaluate the Optuna candidate under later metrics

The Optuna-selected Phase3 candidate achieved the best sampled-validation MAE under global validation metrics.

However, the later Phase4--Phase10 lineage used the manually tuned baseline because it had already been evaluated under region- and profile-level diagnostics.

A useful next step is therefore:

- evaluate the Optuna candidate under the Phase4 system-level metric suite,
- evaluate it under the historical sample491 profile reference,
- compare whether its lower validation MAE transfers to depth-dose and lateral-dose behavior.

---

## 2. Run full strict test2 evaluation

The current strict validation/test summaries use stratified subsets with 300 records per case.

When computational time is available, run a full strict test2 evaluation over the full 4000-record held-out test pool.

---

## 3. Evaluate Phase10D-strict under Phase4 system-level metrics

Phase10D-strict is currently validated mainly through profile-level metrics.

To connect it to the earlier Phase4 system-level evaluation, evaluate it under:

- overall MAE,
- weighted MSE,
- high-dose MAE,
- outside MAE,
- peak-core MAE,
- peak-shoulder MAE.

---

## 4. Perform hard-falloff cohort analysis

The current remaining bottleneck is hard along-beam falloff-shape error.

Analyze records with high falloff peak-normalized error and check whether they share:

- similar geometry,
- similar valid-profile length,
- similar beam-depth behavior,
- specific case IDs,
- low-dose denominator artifacts versus true falloff-shape failures.

<p align="center">
  <img src="assets/tail_uplift_diagnosis.png" width="820">
</p>

<p align="center">
  <em>The key remaining question is whether hard falloff failures form a consistent cohort that can be targeted directly.</em>
</p>

---

## 5. Design falloff-shape-specific refinement

Do not add another generic refinement head immediately.

First confirm the hard-falloff cohort pattern, then design a targeted method such as:

- falloff peak-normalized loss,
- falloff area-under-curve loss,
- asymmetric under-dose penalty,
- improved beam-axis or WED features,
- hard-falloff cohort weighting.

---

## 6. Fully strict end-to-end rerun if needed

Phase10D-strict is a strict refinement-head rerun, not a fully strict end-to-end retraining.

A fully strict rerun would require:

1. training the base model only on strict train6,
2. selecting checkpoints only on strict val2,
3. evaluating once on strict test2,
4. ideally validating on new unseen cases.
