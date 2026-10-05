# Suggested spoken summary

This stage tested whether we could improve the use of a learned WFR magnitude
without retraining the transport model. The initial real-data WFR pilot reduced
whole-cube dose error, but its main array-x profile became worse. A fixed-magnitude
oracle suggested that some of this gap could be addressed by the coefficient
head, although that oracle was target-informed and not a trainable guarantee.

I froze the velocity and growth networks, cached their magnitudes, and trained
two coefficient heads from the same checkpoint. Both received CT and the same
calibrated baseline prediction. One head used direct product supervision; the
other added a relative and absolute x-profile loss. They used the same 192
training records, batch sequence and 384-update budget. A constrained profile
selection rule was fixed before evaluation.

Both selected refinements were accepted by that development rule. The
profile-supervised model reduced mean x-profile percentage error from 8.350% for
the parent to 7.727%, while whole-cube RMSE changed by only +0.028%. Compared with
the composition-only arm, the additional profile loss reduced the x metric by
3.06%, with a small global-error increase. The direction was consistent in both
case means and in the equal-budget last-checkpoint comparison.

The result now improves the listed global and x measures over the saved direct
signed-flow baseline, but it still does not match the final Practical system's
x-profile result of 6.674%. This is one paired study on two reused development
cases, not an independent patient test or a water-reference experiment.

My next step is to check paired refinement-order stability. After that, one
candidate is to give the coefficient head the predicted magnitude explicitly,
because it is choosing how much of that magnitude to use. I would test that
separately from any amplitude repair or change to the transport model.
