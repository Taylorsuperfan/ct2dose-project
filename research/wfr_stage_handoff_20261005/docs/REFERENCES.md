# Sources and attribution

## Source-derived project evidence

The main numeric source is the user's final frozen-magnitude A/B Markdown report.
Its hash, retained precision and exclusions are recorded in
`../provenance/reported_results.json`. The three supplied plots are retained
unchanged. The actual paired protocol is in
`../source/wfr_coefficient_refinement/docs/PROTOCOL.md`.
The parent and older-method contracts are in the corresponding source `docs`.
The supervisor's transcript and private reference PDF are not redistributed.
They motivate studying unbalanced transport, then sign learning and composition;
they do not prescribe this particular architecture or numerical budget.

## Primary mathematical references

- Peng et al. *WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport*.
  arXiv:2601.06810v2, sections 3--4.
  https://arxiv.org/html/2601.06810v2
  Author repository: https://github.com/QiangweiPeng/WFR-FM
  Previously inspected commit: 11c7ae99746ab023f4065f4bfea74484ce651d87.
- Liu, Gong and Liu. *Flow Straight and Fast: Learning to Generate and Transfer
  Data with Rectified Flow*. arXiv:2209.03003.
  https://arxiv.org/abs/2209.03003
- The project's prior method contract explains the spatial Hahn--Jordan
  implementation, zero branches, learned masses and endpoint losses. The
  algebraic positive/negative decomposition should not be confused with a
  theorem that its learned optimization is necessarily unstable.

## Git workflow documentation checked for this handoff

- https://git-scm.com/docs/git-push
- https://git-scm.com/docs/git-switch
- https://git-scm.com/docs/gitignore
- https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request

These sources describe mathematics or tool behavior. They do not supply the
user's reported experimental performance or authorize public data release.
