# Phase9–Phase11 Summary

## Current best result

The current best profile-refinement result is **Phase10D falloff-aware direction refinement**.

On Card17-style sample491:

| Method | Along-beam x | Perpendicular y |
|---|---:|---:|
| Card17 Phase4 mixed weighted reference | 2.775% | 1.615% |
| Phase9G deployable | 4.969% | 0.774% |
| Phase10D falloff-aware | 4.363% | 0.771% |

Phase10D reduces the remaining along-beam error while preserving the perpendicular advantage.

## Important clarification

Card31 was the previous final Phase4 system-level summary using global metrics.
Card17 was an earlier representative-case profile analysis.
The current Phase9/10 pipeline uses Phase5E along-falloff as its actual input base.

## Timeline

- Phase4: baseline / mixed weighted / shoulder-aware / router system.
- Phase5: along-beam / falloff-aware training attempt.
- Phase6–8: representative case, region, and geometry diagnostics.
- Phase9: structure-aware and multiplicative-additive refinement.
- Phase9G: best deployable professor-style profile result.
- Phase10: remaining along-beam vs perpendicular trade-off.
- Phase10D: current best falloff-aware direction refinement.
- Phase10E / Phase11A: ablations that did not beat Phase10D.

## Correct inference note

The corrected rectified-flow sampler starts from CT, not zeros:

- init = CT
- t_mode = midpoint
- sign = +
- euler_steps = 10 for Phase9D-plus / Phase9G reproduction

## Recommended next step

Evaluate Phase10D using the same global/system metrics as Card31:

- overall MAE
- weighted MSE
- high-dose MAE
- outside MAE
- peak-core MAE
- peak-shoulder MAE
