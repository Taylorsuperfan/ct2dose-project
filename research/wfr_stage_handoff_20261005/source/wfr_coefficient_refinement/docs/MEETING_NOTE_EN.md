# Next-stage meeting note

The current model improves whole-cube dose error, but its main array-x profile error remains worse than the final Practical system. A saved-factor review suggests that the present magnitude contains useful information that the coefficient network is not yet using well. The fixed-magnitude oracle is an optimistic reference, not a performance claim.

I am therefore testing two coefficient refinements while keeping WFR transport unchanged. The control learns the actual signed correction with a small auxiliary sign loss. The second arm adds explicit profile supervision. They share initialization, data, batches, optimizer and the extra update budget. Their checkpoints are selected using the same profile-first rule with declared safeguards on other metrics.

Results to fill in after the experiment:

- Did either arm produce an eligible checkpoint, or was the parent retained?
- How much did profile supervision add beyond product supervision?
- Were whole-cube and lateral changes consistent in both development cases?
- How did the selected results compare with Phase10D-strict and the older correction methods?
- Does the evidence favor more work on the coefficient, or is local magnitude coverage now limiting?

The broader research opportunity is to connect the attainable correction envelope to task-aware optimization, and eventually to a coherent signed transport model. Those ideas are not implemented by this experiment, and their novelty remains to be established.
