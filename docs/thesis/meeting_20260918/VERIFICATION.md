# Presentation-revision verification

This is not a rerun of the dose-prediction experiment. It does not establish model accuracy, publication permission or statistical significance.

Executed checks:
- Eight software unit tests passed on the actual renderer and input snapshot.
- The input CSV is byte-for-byte identical to the earlier visualization handoff's reported aggregate table.
- All six original method identifiers are retained.
- The seven new figures were rendered as seven PNG/SVG pairs and their file hashes were checked.
- Each plot uses a single axes. Model files were not loaded; no training or inference ran.
- The charts were visually reviewed for text placement. The typography and descriptive labels were revised, not the metric values.

Scope remaining:
- The user's Drive CSV was not accessed again here.
- Trello cards and GitHub were not modified.
- The private model checkpoints and medical arrays were not accessed.

Actual test output:

```text
test_exact_reported_values (test_presentation.PresentationTests.test_exact_reported_values) ... ok
test_mapping_and_no_opaque_plot_names (test_presentation.PresentationTests.test_mapping_and_no_opaque_plot_names) ... ok
test_refuses_changed_scope (test_presentation.PresentationTests.test_refuses_changed_scope) ... ok
test_refuses_nonempty_output (test_presentation.PresentationTests.test_refuses_nonempty_output) ... ok
test_relative_changes (test_presentation.PresentationTests.test_relative_changes) ... ok
test_seven_figures (test_presentation.PresentationTests.test_seven_figures) ... ok
test_six_unchanged_identifiers (test_presentation.PresentationTests.test_six_unchanged_identifiers) ... ok
test_target_table_unchanged_scope (test_presentation.PresentationTests.test_target_table_unchanged_scope) ... ok

----------------------------------------------------------------------
Ran 8 tests in 0.003s

OK
```
