# The next experiment

## Question and scope

Can a better bounded coefficient use the already learned nonnegative magnitude more effectively, and does explicit array-x profile supervision improve the profile beyond product supervision alone?

The principal causal comparison is **profile versus composition**, not a comparison of unrelated retrained systems. The parent WFR pilot, previous positive/negative flow, direct signed residual flow, Phase10D-strict and shared Phase9G predictor remain saved baselines.

A result against an oracle is diagnostic. A result against a saved model is an empirical comparison. Neither provides a clinical guarantee.

## Fixed components

Let `B` be the saved Phase9G prediction in legacy model units, `S` the existing train-fitted residual scale, `u=(D-B)/S`, and `a` the selected WFR model's reconstructed magnitude. The new prediction is

    raw_dose = (B + S * a * c_psi(CT, B / base_scale)) / dose_scale_factor
    c_psi = tanh(logit_psi / 2)

Only the old grid-sign encoder and its output convolution are optimized. Their 77,041 parameters are copied from the selected parent checkpoint. All 84,836 transport/growth parameters stay fixed, as do its source points, Heun steps and trilinear reconstruction. The two input channels stay unchanged. Magnitude is a fixed multiplier, not an extra model input in this experiment.

Cache `a` once on train192 using the parent inference function. Reuse the previously saved magnitudes on monitor40 and validation600. Validation transport is not rerun. Sign initialization is checked against saved parent raw predictions before each fresh arm starts. Ordinary float32 serialization tolerance is recorded; a material mismatch stops the run.

## Losses

All losses use raw corrections before final dose clipping. Let `e=a*c-u`.

- Reconstruction: mean over all voxels of `e^2`.
- Auxiliary sign: the existing amplitude-weighted binary cross-entropy on `abs(u)>0.01`, normalized within the selected batch. No new positive-class weighting is introduced.
- Relative profile: along the target-peak array-x line, mean `abs(S*e)/(abs(D)+1e-8)` over `D >= 0.01*max(line_peak,1e-8)`.
- Absolute profile: mean `e^2` over that full line.

The target peak and mask are training supervision. They are not inference inputs. Array x means the last array axis, not a verified beam direction. The relative term is a fraction, not multiplied by 100 during optimization.

Loss units are frozen using the unchanged initialization on train192 only. Reconstruction and both profile denominators are their mean initial per-record losses. The sign denominator is the aggregate initial weighted BCE. Each is floored at 1e-8 purely to avoid division by zero, and both raw values and floors are recorded. This normalization sets units; it is not validation-driven tuning or gradient balancing.

The normalized objectives are

    L_A = 0.1 * L_sign / k_sign + L_reconstruction / k_reconstruction
    L_profile = 0.5 * L_relative / k_relative + 0.5 * L_absolute / k_absolute
    L_B = L_A + L_profile

The auxiliary sign weight is deliberately smaller than the reconstruction weight: pure binary supervision prefers saturation, whereas a useful coefficient may need a magnitude below one. This is an engineering choice, not a proven optimal trade-off. It is identical in both arms. Denominator normalization does not guarantee comparable gradients.

The two arms each run 384 added AdamW updates, batch size two, learning rate 1e-4, weight decay 1e-5, gradient clipping at five, and refinement seed29. They share one deterministic with-replacement batch schedule. This seed is a refinement-order seed, not a new independently trained WFR seed. Monitor and save intervals are 32 updates. Per-call defaults are 128 updates.

The selected parent had 384 optimizer updates. Each experiment executes another 384. Comparisons must report both counts. A selected checkpoint can be earlier than the last update, but that does not erase computation spent training the entire run.

## Checkpoint rule, fixed before training

The reference is the current saved WFR model on the original monitor40, not the target-informed oracle and not an easier subset.

A candidate must meet all the following rules on both equal-case aggregates and each case individually:

1. Whole-cube RMSE, x-line absolute RMSE, y/z line mean percentage errors, and y/z line absolute RMSE may not exceed the corresponding parent values by more than 1% relative, plus 1e-6 relative numerical slack.
2. Array-x mean percentage error must not worsen in either case beyond 1e-6 percentage points.
3. Among eligible candidates, select the one with the lowest equal-case x mean percentage error. A decrease must exceed 1e-6 percentage points. Ties retain the earlier selection.

The 1% guard is a prospective development tolerance, not a claim of exact non-deterioration, a statistical interval, or a clinical limit. Do not change it after reading either arm's results. A stricter experiment would use its own new protocol before training.

The unchanged parent is the initial fallback. No eligible improvement means `no_eligible_refinement_parent_retained`. In that case the selected primary prediction is exactly the saved parent output, not an unlabelled copy of the last trained state. The fixed-budget last checkpoint is always retained and reported separately, whether it passes or fails the gate.

Phase10D-strict is a benchmark, not an initialization or an inference input. Passing the parent gate does not mean surpassing Phase10D. Compare those results explicitly afterward.

## Evaluation and fair interpretation

Finish both arms and write PAIR_FROZEN.json before new validation600 evaluation. The lock binds selected and last checkpoints, batch schedule, initial weights, parent source and frozen cache. It does not turn the already observed validation cohort into unseen data.

Re-score all saved predictions with the original metric implementation and case-equal aggregation. The report includes the parent WFR, both selected arms, Phase10D-strict, old positive/negative flows, direct residual flow and the shared Phase9G predictor. The two last-state rows are secondary. Oracle outputs are excluded.

A/B differences can be associated with the added profile loss under this controlled setup. Differences from the original WFR also include additional training, changed supervision and changed selection. Differences from HJD do not isolate the contribution of WFR geometry, since histories and computation differ.

Classify outcomes by named measures, not a hidden weighted score. Report raw and clipped errors, both case deltas, positive and negative recalls, and coefficient contraction. A successful development candidate still needs repeated training seeds and genuinely new case-level evidence before a broader claim.
