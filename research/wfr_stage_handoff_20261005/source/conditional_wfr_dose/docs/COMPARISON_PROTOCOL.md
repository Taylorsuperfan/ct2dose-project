# Paired comparison protocol

## Required methods and identities

1. **Previous final Practical dose system**: saved Phase10D-strict predictions,
   using the restored run01_val600. This is not the early Phase3 checkpoint and
   not a newly approximated architecture.
2. **Previous separate positive/negative correction flows**: saved real HJD pilot,
   `pilot_seed17_hjd_rf_eval01`. This is the method considered easier before the
   later WFR/sign decision. It is preserved, not retuned to favor a comparison.
3. **Previous directly learned signed correction**: saved ordinary residual RF,
   `pilot_seed17_residual_rf_eval01`, a useful signed-output control.
4. **Shared calibrated dose predictor**: saved Phase9G, the common starting point.
5. **New conditional WFR magnitude with grid sign**: the primary new method.
6. **New hard grid-sign sensitivity**: secondary only, not selected after viewing
   validation to replace the primary method.

The new and HJD methods both correct frozen Phase9G. The new model is NOT stacked
on Phase10D. No old model is retrained or reconstructed from tensor shapes here.

## Data and selection

Use the existing train192 cache,6 cases x32. One/six-record stages use only train
records, selected by sample identifiers, and start independently from the pilot.
The pilot uses384 optimizer updates, batch2, seed17. Monitor40 uses the same
identifier-hash selection seed732 as the previous pilot,20 per validation case.
Saved weights are selected by equal-case clipped absolute-dose RMSE on monitor40,
matching the previous new-method selection objective. Sign and magnitude objectives
and architectures are different; equal updates do not imply equal FLOPs or equal
model capacity. Old Phase10D has a different training history and targeted losses.

Evaluate the saved selected checkpoint on the same600 validation IDs/order.
Do not use the true correction, total or signs as inference inputs. The monitor40
is included in these600, and historical upstream exposure remains. This is a
DEVELOPMENT evaluation, not a new blind final test or600 independent patients.
All six methods are re-scored from their serialized stored-unit predictions with
one implementation; tiny differences from older pre-serialization tables can occur.

## Metrics

Per record: whole-cube RMSE/MAE, raw/clipped negative fraction, x/y/z line RMSE,
and line mean/max percentage errors. Lines pass through the target peak, retaining
legacy x=W,y=H,z=D index conventions. The percentage mask is target >=1% of its
line peak; epsilon is1e-8 in legacy internal model units. Absolute line RMSE uses
the complete line. Record means are averaged inside each case, then cases equally.
Never pool all voxels and label the result as a mean of record RMSEs.

Separate diagnostics: normalized residual RMSE; nonnegative magnitude spatial MAE
and total error; active sign accuracy; soft-sign contraction; oracle factor errors.
Oracle quantities use targets only after prediction and never enter model rankings.

## What counts as progress?

No post-hoc scalar score hides a trade-off. Report changes relative to Phase10D
and HJD for BOTH whole-cube and profile metrics, and each of the two case deltas.
A candidate can be called better on a named metric if that measured error is lower.
It cannot be called globally superior merely because RMSE falls while profile error
rises. A stronger pilot claim would require improvement in the declared x-profile
objective and no deterioration in declared whole-volume/lateral objectives, followed
by a new evaluation design and repeated seeds. No significance or clinical threshold
is inferred from this two-case pilot.

## Cost and deployment restrictions

Report new trainable parameters, updates, batch, source particles, solver steps,
velocity/growth evaluations, query chunks and timing separately from shared upstream
cost. HJD's two branches and the new single transport-plus-growth representation
are not computationally matched. No accuracy claim is made from parameter count.

Clinical or physically interpreted testing additionally needs validated dose units,
beam/geometry metadata, an approved external or genuinely unseen case protocol,
and the actual agreed reference field. The frozen learned predictor is not renamed
water dose. No treatment decision may be based on this research code.
