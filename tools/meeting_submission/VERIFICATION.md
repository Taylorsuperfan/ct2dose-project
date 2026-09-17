# Submission-tool verification

This is a packaging and publication-assistance update, not a new training run.

## Executed checks

Eleven unittest checks passed in temporary local Git repositories:

1. Complete file hashes, 196 preserved earlier files, and seven latest PNG/SVG pairs.
2. Dry-run import leaves an existing repository unchanged.
3. Import creates missing files; repeated import is idempotent and does not stage anything.
4. A same-path conflict stops the import before new files are copied.
5. A symlink destination is rejected without writing through it.
6. Modified source content is rejected by the read-only checker.
7. All intended Git-index blobs match the submission.
8. Unrelated staged files are rejected.
9. An intended image omitted from the index is detected.
10. A stale staged version is detected even when the working copy has been restored.
11. Identical files already committed are accepted without requiring a fabricated new commit.

The old release checks, including the clean training/visualization notebook payload
checks, are reused after their file hashes are verified. The original model code,
reported numerical table, old releases and descriptive-name figures were not changed.

Run the tests from the unpacked submission folder:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s submission_tests -v
```

These tests do not run a user's model, evaluate medical arrays, open a user's Drive,
change Trello, push GitHub, audit the full prior Git history or grant publication permission.
Only explicitly listed aggregate pictures are included. Matching a manifest is not
an independent security or scientific certification. Import is conflict-checked but
not a transaction against disk failures or another process writing simultaneously.
