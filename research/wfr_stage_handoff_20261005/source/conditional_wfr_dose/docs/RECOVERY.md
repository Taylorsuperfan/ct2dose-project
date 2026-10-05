# Disconnects, GPU changes and immutable history

The new notebook is independent of W0--W10 and S0--S4. Do not re-run old training.
Each new training call defaults to at most128 updates. The total recipe is256 or
384; a paused call is not a failed experiment. Re-run the SAME stage cell to continue.
Coupling banks save after each bank. A currently unfinished bank is recomputed,
not all previously saved banks. Model checkpoints include optimizer, all used
random generators, monitor history, losses and the last completed update.

## Same environment

Restore N0/N1 and needed dependencies. Use the same code and configuration. Re-run
only the unfinished stage. Complete runs are verified and reused, not retrained.
W7 can resume saved prediction records on the same environment. Completed outputs
can be reused after hash checks without claiming recomputation on a new GPU.

## GPU or library changed during an unfinished run

Do NOT edit/delete contract.json and do not replace a GPU name inside it. Either
restore the old environment, or create an explicit child run using the CLI:

```bash
python -m cwfr fork --cache "$CACHE" --plans "$PLANS" --parent "$OLD_RUN" --out "$NEW_RUN" --device cuda
```

This copies a hash-verified last model/optimizer state and records its parent.
CUDA RNG is explicitly reinitialized when the environment changes. Such a fork is
NOT bitwise continuation. Cumulative updates include the parent. Train the child
using the same config and new output path. Updating the current run variable in
the notebook is required; do not unknowingly resume the parent again.

For a deliberate larger total budget, add `--updates 1536` and use a separately
saved matching Config JSON with updates=1536. The current optimizer uses a fixed
learning rate, not a reset cosine schedule. This is a changed-budget learning-curve
experiment and must not be called the384-update budget-matched pilot.

Partial validation inference changing environment uses a NEW evaluation output
name, not a mix of predictions from multiple unrecorded environments.

The file lock prevents duplicate writers on one runtime only. Do not open two
Colab sessions writing to the same Google Drive run. Files are flushed locally
before pointers are replaced, but Google Drive synchronization is not an absolute
transaction guarantee. Keep the last pointer and its referenced checkpoint together.

Optional-backend environment variables from the old WFR task are set before
imports; the new training solver uses NumPy, not POT/JAX GPU buffers. Do not upgrade
working PyTorch/CUDA to fix a missing optional package.
