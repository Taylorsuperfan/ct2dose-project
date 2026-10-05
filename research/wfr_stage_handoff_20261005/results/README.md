# Reported aggregate results

`comparison_reported.csv` preserves the displayed values in the supplied primary
comparison table. `fixed_budget_last_reported.csv` uses the secondary table's
more limited decimal precision. No extra digits were invented.
`compute_scope_reported.csv` transcribes the supplied budget/selection table.
`derived_changes.csv` contains arithmetic changes from the primary values, not
new model evaluations. Negative relative changes mean lower error; x-mean
absolute differences are percentage points, not relative percentages.

Only aggregate outcomes are included. The original report and per-case rows
contain identifiers and remain private. These data have not been recomputed
from actual CT arrays during packaging. Single-seed, repeated-development-case,
unknown-Gy and array-axis limitations apply to every table.
