# Issues and explicit handling

| Issue | Handling | Remaining limitation |
|---|---|---|
| Artificial reference was not the agreed real task | This package uses only frozen Phase9G real predictions | Not water-reference/physical validation |
| Repeated search and task-registry failures | Standalone embedded Notebook, no Friday task helpers | Existing baseline result files must remain available |
| Drive small-file I/O per batch | Read hash-checked cache to RAM once | First read and report validation still incur Drive I/O |
| Previous HJD mass/shape ambiguity | Save raw components, masses, cancellation and residual signs | Failure mechanism still requires observed real outputs |
| HJD training/evaluation numerical mismatch | Endpoint auxiliary training and evaluation both4096particles/8Euler | Finite-particle error remains |
| Train scale too small/validation tuning | One train-only residualRMS, no validation normalization fitting | Only192records in this firstpilot |
| Notebook disconnection | Model,optimizer,RNG,nextupdate,last/best pointers every16updates | Unsaved updates may be replayed; cloud sync not guaranteed |
| CUDA non-deterministic warnings | Record flags and environment; no bitwiseCUDA claim | Formal conclusions need broader runs |
| Old head has different training budget/exposure | Keep old baseline unchanged and disclose | Controlled development comparison, not end-to-end blind evidence |
| Old axis naming inconsistency | Preserve legacy evaluation, label arrayaxes notbeam/mm | Physical geometry still requires separate evidence |
