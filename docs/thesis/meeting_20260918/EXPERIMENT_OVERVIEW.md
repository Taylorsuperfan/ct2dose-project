# Can learning the remaining error improve our previous dose-prediction system?

Both newly trained methods use the same frozen calibrated dose predictor. Their training target is the supplied ground-truth dose minus this existing prediction. Here, “ground truth” means the paired target dose array, not a claim of measurement without uncertainty. The shared starting prediction is learned from data; it is not a dose-to-water field. This is a controlled error-correction experiment, not a masked water-dose experiment.

## What is being compared?

| Display name | What the method does |
|---|---|
| Shared calibrated dose predictor | Uses the existing calibrated prediction without another correction. |
| Previous final dose-prediction system | Applies the final refinement method developed during the master practical. |
| New: directly learned signed correction | Uses rectified flow matching to learn a single correction field containing positive and negative values. |
| New: separate positive/negative corrections | Uses rectified flow matching with Hahn–Jordan decomposition to learn the two correction components separately and reconstruct their difference. |

The two new methods are alternatives to the previous final refinement. They do **not** apply another correction after the previous final system. The two earlier intermediate models remain in the full result table for context.

## How do the two new approaches differ?

Write the fixed starting prediction as B(C), where C is the computed-tomography (CT) input. The desired correction is r = D_target − B(C).

The direct approach predicts the signed field r. The separated approach defines r_positive = max(r, 0) and r_negative = max(−r, 0). Both component magnitudes are nonnegative. Positive corrections raise an underprediction; the negative component's magnitude is subtracted to reduce an overprediction. An overlap penalty discourages the two predicted components from unnecessarily occupying the same locations; it does not establish exact disjointness by itself.

The term **Hahn–Jordan decomposition** names this positive/negative splitting. **Rectified flow** names the underlying flow model, and **flow matching** the training formulation. The separated model also predicts each component's total magnitude; these voxel sums are not physical mass or deposited energy. The implementational details and terminology mapping are in METHOD_NAMES_CN_EN.md and the unchanged training package's METHOD.md.

## What was completed?

The two new models were trained on 192 cubes from six training cases, with 384 optimizer updates per model and batch size two. Each used one training seed. Their saved model weights were selected by whole-cube root-mean-square dose error on 40 validation cubes. They and the previous system were compared on the same 600 cubes from two validation cases. The 40 selection cubes are included in those 600 cubes, so these are development results, not an independent final test.

Root-mean-square error (RMSE) measures the square root of the mean squared difference. Mean absolute error (MAE) measures the mean absolute difference. The reported whole-cube metrics are first computed per cube, then averaged within each case and equally across cases; they are not one pooled error over all voxels.

A **dose profile** here is the sequence of dose values along an array-aligned line through the ground-truth peak. Whole-line absolute error and local percentage error are different measures. The percentage measure excludes positions below 1% of the target line's peak. These array directions have not been independently verified as physical beam directions.

## What did we find?

Compared with the previous final system, directly learning a signed correction reduced whole-cube RMSE by approximately 4.20%, but increased absolute x-profile RMSE by 3.45%. Learning positive and negative corrections separately reduced whole-cube RMSE by approximately 3.47%, but increased absolute x-profile RMSE by 2.34%.

The corresponding mean x-profile percentage error was 6.674% for the previous final system, 7.895% for the new direct approach and 7.269% for the new separated approach. The separated approach therefore reduced the x-profile regression relative to the direct approach, but did not surpass the previous final system on that objective.

Both new approaches have slightly higher aggregate absolute profile RMSE in all three array directions than the previous final system. There is no overall winner across all reported measures. The current comparison demonstrates a whole-volume/profile trade-off, not general superiority or clinical readiness.

## What comes next?

Before another training experiment, specify the main dose-profile objective and the acceptable changes in whole-cube and other-direction errors. Align both the training losses and saved-model selection rule with that objective. Keep the completed experiment unchanged. This next experiment is planned, not completed, and improvement is not guaranteed.

## Evidence and limits

The numerical table is copied byte-for-byte from the previous presentation package's user-reported, high-precision aggregate snapshot. No new model inference or statistical estimation was performed to prepare this wording revision. The private saved runs remain the primary execution evidence. Historical upstream training exposure, uncertain patient grouping, stored numerical units and CUDA nondeterminism remain disclosed. No confidence intervals or statistical-significance claims are supplied.
