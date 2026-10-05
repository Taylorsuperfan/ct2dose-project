# Profile-aware coefficient refinement over a frozen WFR magnitude

The first real-data WFR pilot improved whole-cube error but worsened the main dose-line measure. A target-informed check then showed that the current magnitude permits substantially better corrections than the learned coefficient produces. This experiment asks a narrow question: **does supervising the actual correction, and then its dose profile, recover part of that opportunity without changing the transport model?**

There are two training arms. Both use the same saved coefficient initialization, frozen magnitudes, 192 training records, batch sequence, optimizer, and 384 extra updates. The `composition` arm learns the signed product with a small auxiliary sign loss. The `profile` arm adds one profile loss. Both use the same predeclared checkpoint-selection rule.

The output is a bounded signed coefficient, not a calibrated probability and not necessarily a pure sign. The WFR velocity, growth, source grid, ODE solver and grid reconstruction remain unchanged. This is a supervised refinement stage over a WFR representation, not a new proof of a signed WFR geodesic.

Open **wfr_coefficient_refinement_colab.ipynb** and follow R0–R10. The notebook contains the complete local code payload, including a byte-identical copy of the earlier `cwfr` package. It does not execute the old training pipeline.

Read these documents before starting:

- [Runbook](docs/RUNBOOK.md): exact cells, expected files, and what to do after disconnection.
- [Experiment protocol](docs/PROTOCOL.md): losses, controls, checkpoint rules, and comparison scope.
- [Current problems and research ideas](docs/RESEARCH_NOTES.md): evidence, mathematical deductions, and untested proposals.
- [Recovery](docs/RECOVERY.md): safe resume and explicit environment forks.
- [Seven-day plan](docs/SEVEN_DAY_PLAN.md): a practical work schedule.
- [Verification](verification/VERIFICATION.md): what was actually tested locally.

## What is deliberately not included

There is no new water-dose reference, clinical dose unit, beam-direction inference, teacher prediction in the model input, automatic hyperparameter search, or test-set evaluation. Known target factors may be used as diagnostics in earlier work; they are not input features or training labels from monitor40 here. Old baseline predictions remain unchanged.

The current 600-cube set is a reused development cohort from two cases. It includes the 40 model-selection records. The parent checkpoint was itself selected on this cohort's development subset. Repeated use of these cases is not independent validation.

If no new checkpoint passes the selection gate, the main result retains the parent. The fixed-budget last state is still evaluated as a declared secondary result, so a failed selection is visible rather than hidden.

## Local command-line setup

```bash
export PYTHONPATH="$PWD:$PWD/parent_source${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -v
python -m refine --help
```

`requirements.txt` lists dependencies. In Colab, retain the working PyTorch/CUDA build; the notebook installs only missing small dependencies. Keep one writer per Drive experiment. All caches, checkpoints and prediction archives belong in private storage, not in a public Git repository.
