# Verification actually executed

## Environment

- Python3.13.5
- PyTorch2.10.0+cpu; CUDA unavailable
- NumPy2.3.5, pandas2.2.3, SciPy1.17.0

## Executed checks

1. **56 CPU tests passed**, with no skips, in10.368 seconds on this machine.
   See `test_log.txt`.
2. The embedded notebook payload was extracted into a fresh temporary directory;
   the SAME56 tests passed again in11.632 seconds. This is not112 distinct tests.
   See `notebook_payload_tests.log`.
3. Notebook schema and all code-cell syntax were validated.
4. Code, configuration and document text was scanned for unintended non-English
   Han-script text; none was found.
5. Four importer checks passed: dry run leaves the repository unchanged, new files
   import, identical files are retained, conflicting files stop without overwrite.

## Test coverage

Analytic WFR endpoint/midpoint/derivative cases; one-point OET solutions; an
independent small convex numerical objective reference; semi-coupling marginals;
unequal target mass preservation; zero-target/cutoff refusal; trilinear totals and
boundary handling; target-free model signature and backward computation; source
quadrature/chunk behavior; conditional weight growth; metric axes and epsilon;
equal-case aggregation; lazy train/validation cache access; data tampering checks;
one/six/pilot selections; actual small training and checkpoint load; bitwise CPU
agreement of interrupted versus uninterrupted small training; explicit fork; two
synthetic-record model evaluation/reuse; and a complete600-record, six-method
comparison using fabricated, explicitly synthetic schema fixtures.

The synthetic test fixtures are NOT measurements from the user's CT dataset.
The comparison fixture contains manually generated prediction arrays for software
checks; its numeric rankings have no scientific meaning.

## Not executed here

- No user medical arrays or trained user checkpoints were available or opened.
- No real one-record, six-record or train192 conditional training was run here.
- No CUDA training, CUDA timing or CUDA determinism was tested.
- No new real validation600 inference or real model comparison was performed.
- The optional comparison to the installed POT library was NOT executed: POT was
  absent and package download failed because network name resolution was unavailable.
  The independent small numerical objective tests above did execute. The optional
  POT test can run separately in the user's connected Colab.
- No exact full-grid UOT optimum, physical transport model, clinical safety,
  superior prediction performance or patient-level independent test is claimed.

## Important remaining scientific risks

Small finite bank supports and finite solver iterations are approximations. The
box-tangent velocity parameterization, source measure, sign loss, trilinear-grid
composition and training budget are proposed choices. They may underfit or fail
to improve profiles. CPU tests cannot establish actual-data learning or guarantee
cross-environment bitwise recovery. Use the declared staged experiment and retain
negative as well as positive outcomes.
