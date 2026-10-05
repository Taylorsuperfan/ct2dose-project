# Resuming after a disconnect

## Ordinary reconnect

Run R0 and R1. Restore missing dependencies in R2 without upgrading the existing numerical stack. Run the unfinished stage, not every prior training stage.

- R4 reuses sealed frozen records.
- R5/R6 restore coefficient weights, optimizer, random states, update count, history and selection state.
- R8 reuses completed prediction records.
- Completed stages verify their receipts and return without repeating inference or training.

Keep the existing work directory, parent checkpoint and registered configuration. Do not delete contract.json, initial weights, normalizers or checkpoint pointers.

## GPU or numerical environment changes

A different GPU name or library version is not proof that data are invalid, but it is not an exact continuation either. Cache preparation can be continued explicitly with `--allow-cache-environment-change`; each new record stores its environment and an event is retained. It does not overwrite sealed records or claim bitwise equality.

For unfinished training, create a child run. For example, after restoring the notebook variables:

```python
task(
    "fork",
    "--work", WORK,
    "--parent-run", COMPOSITION_RUN,
    "--out", WORK / "runs.local" / "composition_seed29_gpu_change",
    "--device", DEVICE,
)
COMPOSITION_RUN = WORK / "runs.local" / "composition_seed29_gpu_change"
```

Then rerun R5. Use the corresponding profile path and R6 when applicable. The child inherits the same total384-update refinement budget, optimizer, saved selection and CPU/NumPy random state. CUDA random state is explicitly reset on environment change. This is marked as a non-bitwise continuation, not a new independent seed. Batch order is still fixed by the registered schedule.

If either arm used a different environment, the pair lock reports that difference. Prefer the same device and stack for both arms when practical. The default notebook paths are restored by R1, so after reconnecting, restore any chosen child-run variables before resuming or finalizing.

A completed arm is not forked as an unfinished run. A longer training budget is a different experiment and is deliberately not enabled as a casual resume option here.

For a partial validation evaluation under a different environment, use a new evaluation directory and set EVALUATION accordingly before R8/R9. Do not combine records from different unrecorded runs.

## Abrupt interruption during a file write

Writes use a temporary file followed by rename, with array and checkpoint hashes. A deliberately paused run is resumable. There is still a short gap between saving an array and saving its receipt. An unsealed archive is not silently trusted or replaced: the program stops and names it. Preserve it and inspect the interrupted stage rather than deleting the whole experiment.

Checkpoint selection receipts are recovered from the latest checkpoint when that checkpoint contains the selected state. If the required selected state cannot be identified safely, the program stops. The last successfully sealed checkpoint, not the last line printed on screen, defines recoverable progress.

The local writer lock prevents two local processes from writing the same run. It is not a distributed lock across separate Colab machines. Use only one active runtime per experiment directory.

## Reproducibility boundary

CPU tests check exact interrupted-versus-uninterrupted optimizer results in the tested environment. No CUDA or cross-version bitwise guarantee is made. The reused parent source, selected checkpoint, cache and protocol are checked separately from the runtime environment.
