# Decision log update — 2026-09-17

**Status-at-result review; append as a new note. Do not overwrite historical decisions.**

Observation: old Phase10D-strict source/weights were recovered and val600 evaluated. A real Phase9G-error pilot trained ordinary residual RF and HJD and produced six-method comparison results on the same val600 cohort. Both new models reduce global RMSE/MAE but increase x-profile percentage and absolute RMSE versus Phase10D-strict.

Interpretation: the current implementation learns useful global corrections. Training/selection primarily target global error and do not directly enforce profile non-regression; this is an explanation supported by code, not proof of the sole cause. HJD's representation is not established as universally superior.

Limitations: one seed; two validation cases; 40-record monitor overlaps val600; unequal historical/new budgets; inherited upstream exposure; legacy feature/profile axis mismatch; unknown physical-unit calibration; CUDA nondeterministic operations. Not water dose or blind final testing.

Decisions: freeze V1 source and results; keep old checkpoints and all raw evaluation results unchanged. Publish source and permitted aggregate summaries only after review. Keep histories, manifests, record-level tables, images and model weights in private storage. Do not change the baseline or select another checkpoint after observing val600 to make V1 win.

Next test (not launched): design V2 with explicit reconstructed-dose profile supervision and a declared checkpoint-selection rule for both new methods. Specify axis conventions, monitored metrics, guardrails, weights and seeds before optimization. Preserve V1 as the comparison reference. Source/target alternatives requiring advisor input remain separate.
