# Verification performed for this release

## Executed locally

The final source suite passed **43 CPU tests** with no skipped tests. The test environment used Python3.13.5, PyTorch2.10.0+cpu, NumPy2.3.5 and Pandas2.2.3. The full log is in `test_log.txt`.

The tests include:

- the same coefficient architecture and exact copying of parent coefficient weights;
- no mutation of parent transport parameters during coefficient optimization;
- bounded outputs, the target-free prediction signature and one actual small-grid parent inference;
- raw product and profile losses, axis selection, masks, finite gradients and zero-target loss handling;
- identical batch schedules, strict A/B loss difference and per-case selection guards;
- non-overwriting writes, hash failures and changed-protocol rejection;
- cache preparation on a complete manufactured train192/validation600 schema;
- actual coefficient optimization on that fixture, checkpoint save/load, exact CPU interrupted-versus-uninterrupted parameter comparison, explicit forks and environment mismatch rejection;
- completed paired coefficient evaluation and old-method comparison on600 manufactured validation records, plus plot and report generation;
- reuse of completed stages without model inference.

The schema fixture marks its parent completion as a fixture, not as384 genuinely trained parent updates. One32-cube cache entry used the actual unchanged parent `predict` function. To keep the remainder of the cache test bounded, repeated parent trajectory inference was replaced with a clearly labelled identity-magnitude test substitute. The coefficient training, checkpoint mechanics,600-record coefficient inference and comparison still ran through the actual implementation. These tests do not demonstrate medical accuracy or CUDA performance.

The unchanged parent `cwfr` source bytes were compared with `conditional_wfr_dose.zip`; every bundled parent module matched. `docs/PARENT_SOURCE.json` records the archive and module hashes.

The notebook format and code-cell syntax are checked at construction. Its embedded code is unpacked into a fresh directory and tested separately; that repeat is the same43-test suite, not43 additional test designs. Installer checks use a temporary Git repository and do not write to the user's repository.

## Not executed here

No user's Google Drive cache, medical array, trained checkpoint or new refinement result was available for execution. No CUDA experiment, new validation improvement, successful checkpoint gate, GPU runtime, clinical dose calibration, permission review, GitHub push or Trello change is claimed.

No original WFR author-reproduction run or old Practical recovery is rerun by this release. Existing empirical numbers in the research notes come from the supplied project logs, not from the tests.

The new hyperparameters are explicit engineering choices. The code is executable research software, not a guarantee that either arm will improve the requested objectives.
