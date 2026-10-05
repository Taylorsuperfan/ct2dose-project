# Next work: consolidate one finding before changing the model

## Now: archive the completed pair

Retain the selected A/B outputs and both 384-update final states. Keep the same
reported cohort, units and metric implementation. The selected steps are 32 and
160, but both arms executed 384 extra updates. Nothing in this handoff starts
training, changes the old output directory or promotes a secondary checkpoint.

## Next experiment: paired refinement-order repeats

Write a new protocol before running two additional A/B pairs with predeclared
refinement seeds. Seeds 37 and 53 are reasonable reproducible labels, not
optimized values. Keep the original parent checkpoint, fixed magnitudes,
train192, monitor40, network, optimizer, losses, 384-update budget and selection
rule unchanged. Each pair shares its batch order. Start both arms from the
original parent, not from the successful profile-selected head.

The question is sensitivity to the refinement batch order, not independent WFR
training, patient generalization or a larger test set. Report every planned seed
and failures; do not select the best seed. New work roots and registered
configurations are required because the existing workspace binds its seed.
The source recovery instructions explain immutable contracts and explicit forks;
an interrupted run resumed after a hardware change is not a new independent seed.

## Then: test one architectural hypothesis

A bounded coefficient must choose how much of the predicted amplitude to use,
yet its current inputs are CT and B(C), not amplitude itself. Test an explicit
magnitude-conditioned head c(C,B,a) against the current c(C,B) under a matched
budget. Since a is itself a deterministic prediction from the same inputs, this
does not add target information; it may make a useful intermediate computation
more accessible. It is not an established gain and is not implemented in this
release. A no-new-channel control and a matched-capacity control would help
separate the effect of the input from parameter-count changes.

Do not simultaneously change reconstruction, losses, width, training set and
transport fields. The prior 40-record fixed-amplitude oracle remains target-informed
evidence and cannot be treated as a deployable performance goal on 600 records.

## Later if needed: repair local magnitude, not only the global total

Where |u|>a, a bounded coefficient has a hard pointwise limitation. A future
local amplitude repair could be constrained with a square-root amplitude
penalty motivated by reaction geometry. This is a hypothesis, not a new
validated WFR metric. Positivity, zero support, physical/numerical units and
identifiability need explicit definitions. Never calibrate predictions using
a validation target total.

## Independent evidence and the physical application

Plan genuinely new case-level evaluation after freezing the full chain, not just
the final head. Audit historical patient grouping and upstream model exposure.
Do not turn the remaining 560 already inspected cubes into a new untouched test.
Confirm the intended water/reference-field definition, geometry and beam
metadata with the supervisor. A change from learned-baseline residuals to an
actual water-referenced residual is a separately defined target and experiment.

## Suggested short schedule

Day 1: import and review this handoff; do not rerun completed training.
Day 2: register the paired-repeat protocol and separate output roots.
Days 3--4: execute both fixed-budget pairs; inspect only the designated monitors.
Day 5: freeze all choices, evaluate and report all repeats.
Day 6: summarize effect directions and runtime, separating parent and head costs.
Day 7: decide whether the magnitude-input hypothesis warrants one new controlled
experiment, and document independent-data and reference-field requirements.

None of the repeat or architecture experiments above is claimed completed.
