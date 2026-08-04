# Risk Register

| ID | Risk | Probability | Impact | Detection | Mitigation | Status |
|---|---|---:|---:|---|---|---|
| R1 | Beam metadata missing or inconsistent | High | High | Metadata audit | Clarify data source before physics implementation | Open |
| R2 | Only normalized CT available | Medium | High | Compare raw and processed arrays | Recover raw HU or limit physical claims | Open |
| R3 | Cube omits upstream beam path | Medium | High | Inspect full-volume geometry | Compute physical prior on full volume before cropping | Open |
| R4 | Physical coefficients are underdetermined | High | Medium | Phantom and validation sensitivity tests | Calibrate on train/val only and report uncertainty | Open |
| R5 | Too few independent cases | High | High | Case count audit | Obtain more cases or limit generalization claims | Open |
| R6 | Test leakage | Medium | High | Automated split tests | Lock final test access and audit every config | Open |
| R7 | PDE solver instability | Medium | High | Phantom/grid convergence tests | Start with simple stable discretization | Open |
| R8 | CFM does not beat residual regression | Medium | Medium | Matched baseline | Report as valid negative result | Open |
