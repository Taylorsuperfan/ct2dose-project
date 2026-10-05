# Verification — real Phase9G residual pilot tooling

Executed in the assistant's local CPU environment, not the user's Colab/Drive.

## Latest complete test run
```
......................................                                   [100%]
38 passed in 32.72s
PYTEST_RETURN_CODE 0
```
Includes 24 unchanged legacy recovery tests and 14 new training/cache/comparison tests.
Separate runs also returned zero: new14 in4.40s, legacy24 in29.87s.

## Tested behavior
- Thirty exact original function/class segments verified against their source hashes.
- All ten vendored p10recover Python modules are byte-identical to the previously delivered recovery version.
- Signed HJD decomposition, zero branches, target sampler, unit-sum splatting, boundary reporting, finite gradients.
- Dense signed ODE does not clamp negative residual states.
- Both new models pass forward/backward and32³ shape tests on temporary software tensors.
- Normalization uses training arrays only; all-zero correction task is rejected.
- Identifier-only selection and exact current validation cohort/order checking.
- Cache producer receives CT only before target loading; only training records cause upstream prediction calls.
- Saved validation arrays are referenced and read back, not newly inferred or duplicated in the new cache.
- Cache tampering and source/config/checkpoint mismatches rejected.
- Both RF and HJD CPU interrupted-at-update/resume tests match corresponding continuous runs.
- Model/optimizer/RNG/checkpoint hashes persisted and read through weights_only=True.
- End-to-end fresh training, new prediction, paired comparison and report generation on temporary software data.
- Standalone Notebook P0 actually executed: thirty source definitions verified,14newtests passed, exit0.
- Notebook:20cells,9code cells; validated and parsed; no execution outputs embedded in delivered copy.
- Public-tree preflight passed. This is not a guarantee of absence of private data in future runtime outputs.

## Parameters at the default recipe
- ordinary residual RF:340,545 trainable parameters.
- spatial HJD:104,144 trainable parameters.
These exclude the shared frozen old pipeline. Different parameter counts and computation are disclosed.

## Not executed here
No real user checkpoint loading; no user medical-array/cache reading; no new real training;
no real RF/HJD performance result; no new600record GPU evaluation; no GitHub push or Trello mutation.
Software fixtures used for tests are NOT proposed artificial reference data for the actual experiment.

## Reproducibility limitations
CUDA interpolation/pooling/scatter may be nondeterministic despite a saved seed.
The checkpoint continuation guarantee is bounded by saved updates and environment/source/data identity.
Drive synchronization/failure and concurrent writers on different VMs are not controlled by CPU tests.
An earlier combined-test invocation under a slow mounted temporary directory reached38passes
but the outer runner timed out during shutdown; the final local-/tmp invocation above returned0.
