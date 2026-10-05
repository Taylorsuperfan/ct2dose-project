# Running the refinement study in Colab

Use the new notebook. Do not edit the old N0–N9 cells or copy new files into the old `cwfr` directory. The new notebook restores its own code and an unchanged copy of the old parent modules.

## R0 — Restore the code

Run the cell once after a new runtime starts. It extracts the embedded, hash-checked payload to a content-addressed folder under `/content`. It does not mount Drive or open medical data.

## R1 — Mount Drive and set paths

The defaults match the completed project:

```text
Parent work:
/content/drive/MyDrive/ct2dose_thesis/conditional_wfr_dose

Existing cache:
/content/drive/MyDrive/ct2dose_thesis/phase9g_signed_real_v1/cache.local/train192_seed42

New work:
/content/drive/MyDrive/ct2dose_thesis/wfr_coefficient_refinement
```

The old parent must contain its completed pilot checkpoint and validation600 outputs. The old HJD and direct-flow evaluation directories are needed for the final comparison, not for a new reproduction of their training.

R1 defines `task`, `execute`, and the selected device. No token, case label, raw scan path or target array is used as a model feature. Do not use two runtimes to write this new work directory simultaneously.

## R2 — Check dependencies and software

Keep the working PyTorch installation. Missing NumPy, Pandas, Matplotlib or tabulate may be installed; PyTorch is not automatically upgraded. Tests run in a CPU subprocess. Their manufactured results are software fixtures, not new medical measurements. The full log is saved, and only its tail is shown by default.

Run this on the first setup. After a simple runtime disconnect with the same code and a preserved successful test log, repeating the full tests is optional. Restoring a new or changed environment still requires checking compatibility; skipping tests does not bypass contracts.

## R3 — Register the experiment

Read `docs/PROTOCOL.md` and the displayed configuration. Then run R3. This writes one immutable protocol and records the parent checkpoint, source, data selection and both arms.

The 1% relative guard is a research choice, not a clinical tolerance. It is not to be adjusted after seeing the two new results. Change a proposed setting only before registering a new experiment, in a new work directory. Do not edit protocol.json to force a later run to match.

No medical arrays are opened in registration.

## R4 — Build the frozen magnitude cache

Set the flag in this actual cell:

```python
RUN_PREPARE = True
```

The cell processes records in chunks of32. With `RUN_TO_COMPLETION=True`, it starts the next chunk automatically after the previous one returns. A disconnect loses at most the currently unfinished record; sealed records are verified and reused.

For the192 training records, it runs the selected, frozen WFR model once per record. For monitor40 it reuses saved validation magnitudes rather than integrating again. It also stores the original coefficient initialization and fits the loss denominators from train192 only.

Expected completion:

```text
FROZEN CACHE COMPLETE: .../frozen.local
```

The cache contains private conditions, targets and provenance. Do not upload it to GitHub.

The underlying source grid and model are unchanged. This stage requires inference, but no learning. It does not recalculate UOT banks or retrain WFR.

## R5 — Train the composition-supervised control

Set `RUN_COMPOSITION_ARM=True` in R5. The default per-call limit is128 updates and the total is384. `RUN_TO_COMPLETION=True` continues automatically across those three calls. Checkpoints are written every32 updates and at deliberate pauses.

The log prints loss, monitor x percentage error, whole-cube RMSE, eligibility and whether a new checkpoint was selected. `eligible=False` is a result of the declared guard, not a runtime failure.

Do not judge only the weighted loss. Check the actual dose/profile metrics and the recorded positive/negative recalls.

## R6 — Train the profile-supervised arm

Set `RUN_PROFILE_ARM=True` in R6. It starts from the same parent initialization as R5, not from the final R5 model. Its batch schedule, update count and optimizer are identical. The only arm-specific objective term is the profile loss.

Both arms are trained even if one looks better early. Do not stop an arm just because its monitor result is inconvenient.

## R7 — Review both summaries

R7 reads the training summaries. It does not access validation600.

`accepted_refinement` means a new checkpoint met the declared development gate and improved the main monitor objective. `no_eligible_refinement_parent_retained` means none did. That is a valid outcome, not a reason to loosen the guards.

The original parent is also the fallback. The selected step can be0. The fixed-budget last state is preserved as a secondary comparison even when the selected primary state is the parent.

## R8 — Lock the pair and evaluate

After both budgets finish, set `RUN_LOCKED_EVALUATION=True` in R8. The cell writes PAIR_FROZEN.json, then evaluates both selected and both last-state coefficient heads on the same600 records.

This uses the already saved WFR magnitudes. It does not integrate WFR or retrain any old model. The default evaluation chunk is100 records; the loop continues until completion. Partial outputs are reused only under the same evaluation identity and environment.

The fallback row copies the saved parent prediction exactly. It is explicitly labelled as retaining the parent; no new model performance is invented.

## R9 — Compare with saved old methods

Set `RUN_COMPARISON=True`. The old Phase10D-strict, HJD and direct residual-flow predictions are re-scored without retraining. Data IDs, source links and saved bytes are checked.

The primary table contains7 methods. Two fixed-budget last-state rows are written separately. Never substitute those secondary rows into the primary ranking after seeing which is smaller.

## R10 — Read the report

The report includes global RMSE, x percentage error and x absolute line RMSE figures, paired case differences and the extra training budget. Do not ignore y/z columns or raw-versus-clipped differences in the detailed tables.

Read both A-versus-B and the old-baseline comparison. A-versus-B addresses the profile term. Improvement over the parent also includes extra training, product supervision and a different selection rule.

## What to send back

Send the two training_summary.json outputs and comparison.csv. Include fixed_budget_last.csv if no candidate passed the gate. Keep patient identifiers, arrays and provenance ledgers private.

There is no need to rerun the completed WFR simulation, the three-dimensional bridge, or the original Practical recovery before this study.
