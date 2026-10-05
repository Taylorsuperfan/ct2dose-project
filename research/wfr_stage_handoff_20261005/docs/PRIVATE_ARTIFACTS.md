# What stays outside Git

This is a source-and-aggregate-report release, not a full byte-for-byte experiment
archive. The following remain in authorized private storage:

- Original CT/dose arrays, patient/case maps and train/validation manifests.
- The train192 cache and all cached magnitudes/prediction volumes.
- Parent, refined and Practical checkpoints, optimizer states and RNG states.
- Actual training histories, selection pointers and PAIR_FROZEN.json when they
  contain identifiers or private paths. Retain them unchanged in Drive.
- Per-record/per-case output tables and the original unredacted report/transcript.

Recorded Colab workspaces are `ct2dose_thesis/conditional_wfr_dose`,
`ct2dose_thesis/wfr_coefficient_refinement`, the earlier
`phase9g_signed_real_v1/cache.local/train192_seed42` and the recovered Practical
`phase10d_recovered_v1/run01_val600`, all under the user's mounted MyDrive.
No existing data are fetched or changed by the handoff tools.

The new report uses the submitted aggregate values and supplied plots. It does
not claim to have checked the user's full Drive against this public bundle.
`provenance/source_archives.json` identifies the source distributions supplied
in the conversation. A local comparison tool is included to compare the restored
Colab R0 source to those copies without loading weights or arrays.

No individual case identifiers are retained in the release. Aggregate research
results can still require permission. A private repository is not permission to
share medical data. Review the existing repository history as well as new files;
a new branch inherits its ancestor commits. A new .gitignore cannot remove
sensitive content already committed in an ancestor.
