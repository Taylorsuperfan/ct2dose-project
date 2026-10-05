# A practical seven-day plan

These are workdays from the actual start, not a prediction of GPU runtime. The next stage is deliberately one controlled experiment rather than several new models.

| Day | Work | Deliverable |
|---|---|---|
| 1 | Read the protocol, run R0–R3, start R4. Keep the current parent and old comparisons untouched. | Registered A/B plan and verified parent selection. |
| 2 | Finish R4. Check that232 cached records consist of192 training and40 monitor records. Start R5. | Frozen cache, train-only loss scales, first control checkpoints. |
| 3 | Finish the control's384 added updates. Start R6 from the same parent initialization. | Complete control history, not just its lowest loss. |
| 4 | Finish the profile arm. Read R7. Record whether either arm met the guard; do not change tolerances. | Two summaries and a clear accepted/no-eligible outcome. |
| 5 | Lock the pair and finish R8. | Saved selected and last-state predictions on validation600. |
| 6 | Run R9–R10. Inspect per-case changes and the negative results as well as improvements. | Paired comparison, figures, compute accounting. |
| 7 | Write a short English meeting note. Decide whether to replicate, study coefficient capacity, or repair local magnitude. | Evidence-based next decision and a clean source commit. |

After R4, the main training updates only the coefficient network. Magnitude inference is not repeated in every optimization step. A run that fails the guard is still an informative result. A technical error should be resolved before proceeding; a scientific non-improvement should be reported rather than hidden.

The next decision should follow the evidence:

- A and B both improve, with B adding value: replicate the controlled contrast and broaden case-level evaluation.
- A improves but B does not: inspect the profile objective/optimization rather than declaring WFR ineffective.
- Both fail despite reducing training loss: investigate condition sufficiency, capacity, train-to-validation shift and parent exposure.
- Coefficient gains level off near the fixed-magnitude envelope: consider the local magnitude-repair research idea.

Do not implement all future ideas at once. A thoughtful negative result and a controlled follow-up are more useful than an uninterpretable mixture of changes.
