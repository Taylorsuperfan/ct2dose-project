# Files to keep where

| Location | Keep |
|---|---|
| Git working tree: research/phase10d_recovered_comparison | Byte-identical released recovery source, tests, records/provenance and output-free delivery notebook |
| Git working tree: research/phase9g_signed_real_v1 | Byte-identical training, cache/compare code, frozen p10recover dependency, tests, method doc and output-free delivery notebook |
| Git working tree: this update folder | Latest scope, aggregate-only metric transcription, paired aggregate changes, decisions and registry |
| Drive/private backup | All trained weights, optimizer/RNG states, last/best pointers, environment, hashes/contracts, train/val case or patient lists, caches, medical arrays, raw/processed record-level predictions, training logs, medical/profile images and private report HTML |
| Outside repository, private backup | Original output-rich Notebook and supervisor PDF; do not replace them with the clean submission notebooks |

The repeated p10recover directories are intentional frozen dependencies and currently byte-identical. Do not delete or refactor them during this release. Run each package's tests with its own root as the working directory; mixing the two PYTHONPATH roots can import the wrong copy.

The requirements files are dependency lists, not full environment lockfiles. Do not claim identical future environments merely because they are committed. Actual environments remain in the private run contracts; a separately reviewed stripped environment summary may be published later.

Neither Git LFS nor a private repository removes the need to control access to medical data/weights. No extra redistribution license is inferred from this handoff; preserve existing attribution and confirm permission before making unpublished code/results public.
