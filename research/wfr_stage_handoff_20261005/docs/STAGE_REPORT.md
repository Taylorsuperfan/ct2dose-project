# What this stage did

## Motivation

The initial real-data conditional WFR pilot improved whole-cube error but worsened
the main array-x dose-line objective. Its checkpoint rule selected for whole-cube
RMSE, and the profile regression was already visible on the 40 selection records.
The next question was therefore narrower than building another transport model:
can the already learned nonnegative magnitude be used more effectively?

An earlier target-informed diagnostic on that same selection subset provided a
reason to try. With the predicted magnitude fixed, the bounded pointwise oracle
had an x-profile percentage error of about 5.338%, versus 9.876% for the current
raw prediction. The Practical result on that 40-record subset was about 7.978%.
Those are diagnostic subset numbers, not the new 600-record measurements and not
performance reachable by a finite conditional network by guarantee. Positive
sign recall was about 28.0%, negative recall about 88.9%, and the soft coefficient
retained about 39.5% of predicted magnitude. These observations motivated direct
supervision of the product instead of forcing every coefficient to a hard sign.

## Prediction and units

Let C be CT, B(C) the frozen calibrated Phase9G prediction, and D the paired target
in the legacy internal model scale. The target correction is

\[
r=D-B(C),\qquad u=r/S.
\]

S is the existing positive, train-fitted residual RMS scale, approximately
0.0045299765350856275. It is reused, not fitted on validation. The trained WFR
system supplies a reconstructed nonnegative magnitude a. The coefficient head
uses the original two input channels, normalized CT and scaled B(C), and outputs

\[
c_\psi=\tanh(\ell_\psi/2),\qquad \hat u=a c_\psi,
\qquad \hat D_{\mathrm{stored,raw}}=(B(C)+S a c_\psi)/F.
\]

F is the existing stored-unit conversion, not the residual scale S. The primary
historical dose comparison clips final absolute dose at zero; signed corrections
are not clipped to nonnegative. Stored units are not independently verified Gy.

The head is called a **bounded signed coefficient**, not a calibrated probability.
It selects direction and the fraction of the supplied magnitude to use. This
terminology matters because a useful factor may have an absolute value below one.

## What stayed fixed and what changed

All 84,836 parent transport/growth parameters stayed fixed, together with the
source grid, 16 Heun steps and trilinear reconstruction. Only the 77,041 parameters
of the old grid-sign encoder and output convolution were optimized. No extra
input channel, teacher prediction or local gain network was added.

The parent model supplied fixed magnitudes for 192 training records. Magnitudes
on monitor40 and validation600 were reused from the already saved parent
evaluation. Caching involved fixed-model inference, not new transport training.
The reference is Phase9G, not dose-to-water, and neither correction is appended
after Phase10D. Phase10D is a separate saved benchmark.

## Controlled arms

Both arms started from the same selected parent coefficient weights. They shared
the same frozen cache, records, deterministic batch sequence, optimizer settings,
monitor schedule and total extra-update budget. B did not start from A's output.
Let e = a c - u. Both arms use the original amplitude-weighted sign BCE on |u|>0.01
and a whole-grid squared reconstruction term. B additionally uses a relative
x-profile absolute error and an absolute x-profile squared error.

\[
L_A=0.1L_{\mathrm{sign}}/k_{\mathrm{sign}}
+L_{\mathrm{reconstruction}}/k_{\mathrm{reconstruction}},
\]
\[
L_B=L_A+0.5L_{\mathrm{relative\ profile}}/k_{\mathrm{relative}}
+0.5L_{\mathrm{absolute\ profile}}/k_{\mathrm{absolute}}.
\]

The k constants were measured at the unchanged initialization on train192 only,
then frozen with a numerical floor. They set loss units, not validation-tuned
weights or balanced gradients. The relative term uses positions with target dose
at least 1% of the target peak on the selected array-x line. The line and mask
are training supervision, not inputs to deployment. The absolute term uses the
whole line. The source protocol is reproduced without rewriting it in
`source/wfr_coefficient_refinement/docs/PROTOCOL.md`.

Each arm executed 384 extra AdamW updates, batch size 2, learning rate 1e-4,
weight decay 1e-5, gradient clipping 5, refinement seed 29, and monitor/save
interval 32. The parent had 384 optimizer updates. Seed 29 is a refinement-order
seed, not an independently trained WFR parent.

## Checkpoint selection and data discipline

The checkpoint rule was declared before this pair ran. On monitor40, at both the
case and equal-case aggregate levels, whole-cube RMSE, x-line absolute RMSE and
y/z relative and absolute line metrics could worsen by at most 1% relative, plus
the stated small numerical slack. Array-x percentage error could not worsen in
either case. Among eligible candidates, the rule selected the lowest equal-case
x percentage error. The unchanged parent was the fallback.

This is a development safeguard, not a clinical tolerance. It does not promise
non-deterioration on later data, and the gate is against the parent WFR model,
not the best Practical result. A selected-step difference can arise from the
combined constraints and monitor objective; it is not by itself proof of overfit.

A selected additional step of 32 was accepted for A, and 160 for B. Both arms
actually ran 384 steps, and both final fixed-budget states were retained. A lock
bound model choices before the new 600-record coefficient evaluation. Locking
choices does not make an already reused cohort independent.

## Evaluation

All seven main systems were scored from serialized predictions with the same
implementation on 600 cubes from two development cases. The 40 selection records
are included. Metrics were first computed per cube, averaged within each case,
then equally across cases. A reported mean of per-cube RMSEs is not a single
pooled RMSE over every voxel. x/y/z are array axes, not verified beam directions.

The public-oriented aggregate tables here were extracted from the supplied final
report; the original case IDs and per-case table were deliberately excluded.
The original report states that the old Practical, positive/negative and direct
residual predictions were unchanged. This handoff itself has no private arrays
with which to repeat those model-level computations.

## Primary results

| System | Whole-cube RMSE (stored units) | x mean percentage error | x absolute RMSE |
|---|---:|---:|---:|
| Parent WFR | 4.0439566e-6 | 8.3497114% | 1.4458024e-5 |
| A: composition supervision | 4.0412911e-6 | 7.9706312% | 1.4323579e-5 |
| B: added profile supervision | 4.0450964e-6 | 7.7271259% | 1.4204694e-5 |
| Previous direct signed flow | 4.1457718e-6 | 7.8950025% | 1.4335893e-5 |
| Previous positive/negative flows | 4.1773838e-6 | 7.2686189% | 1.4181766e-5 |
| Final Practical system | 4.3277492e-6 | 6.6741685% | 1.3857424e-5 |
| Shared calibrated predictor | 4.3255640e-6 | 7.0215464% | 1.4021039e-5 |

Relative to A, B reduces x mean percentage error by about 3.055% and absolute
x-profile RMSE by 0.830%, with a 0.094% whole-cube RMSE increase and a 0.805%
MAE increase. Relative to the parent, B reduces x percentage error by 7.456%
and x absolute RMSE by 1.752%, while whole-cube RMSE rises by 0.028%. These small
global differences are numerical observations, not demonstrated statistical
equivalence. The profile improvements occur in both case-level means.

B improves the listed global and x-profile measures over the saved direct
signed-flow baseline. Against the previous positive/negative method, global
RMSE is lower by about 3.17%, but x percentage error is higher by about 6.31%.
Against the final Practical system, global RMSE is lower by about 6.53% and MAE
by 10.95%, while x percentage error is higher by 15.78% and x absolute RMSE by
2.51%. No overall superiority claim is supported.

At the equal 384-update final states, B also has lower x percentage error than A
(7.69346% versus 8.23864%) and a slightly higher global RMSE. This secondary result
supports a changed optimization trade-off rather than a selected-step accident.
It does not replace the primary selected checkpoints after looking at 600 records.

## Interpretation and limits

The result supports a useful supervised refinement of a fixed WFR representation.
It does not show that WFR transport itself improved during this stage. It also
does not isolate WFR geometry from differing architectures, histories and compute
budgets of the older systems. The two-branch method was not proven unstable;
its main x-profile performance remains better here.

The remaining gap could involve coefficient approximation, loss conflicts,
local amplitude undercoverage, or other mechanisms. This aggregate table does
not uniquely identify them. In particular, the fixed-amplitude envelope remains
[-a,a], but the achieved head may still be far from its pointwise oracle.
The earlier 40-record oracle must not be subtracted from the current 600-record
metric as if they were the same evaluation population.

This is one paired refinement run on one parent and a small reused development
cohort. Publication of these aggregate data still needs appropriate permission.
Independent evaluation must audit exposure through every upstream component.

## What comes next

Freeze and archive this stage. Run two predeclared paired refinement repeats
with changed batch-order seeds and the original parent fixed. Report all repeats,
not their winner. Only then test one architectural change, such as explicitly
conditioning the coefficient on the already predicted magnitude. If local
amplitude undercoverage remains limiting, investigate a constrained amplitude
repair in a separate experiment. None of these proposed experiments is recorded
as completed in this handoff.
