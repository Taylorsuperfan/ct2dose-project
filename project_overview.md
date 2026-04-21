# CT-to-Dose Prediction: Project Overview

This document provides a phase-by-phase overview of the CT-to-dose prediction project on paired `32×32×32` CT/dose cubes.

For a shorter repository homepage summary, see the main [`README.md`](README.md).

---

## Overview

This project studies the mapping from paired CT cubes to dose cubes using regression-based and flow-based models:

`CT cube → dose cube`

The project evolved through four main stages:

- **Phase 1** — pilot experiments in 2D and 3D
- **Phase 2** — analysis and interpretation of pilot results
- **Phase 3** — larger-scale 3D flow-model development and tuning
- **Phase 4** — targeted evaluation, subset analysis, routing analysis, and refinement experiments

The current focus is on **clinically relevant high-dose structure**, rather than treating all spatial regions as equally important.

---

## Project goal

The main goal is to learn a stable and accurate CT-to-dose mapping while understanding:

- how well the beam-shaped dose structure is reconstructed
- where different models succeed or fail
- which model is best under different evaluation priorities
- whether targeted refinement can improve high-dose and transition-region quality

At the current stage, the priority is:

1. high-dose region quality  
2. structure-sensitive region quality  
3. transition / shoulder-region accuracy  
4. global stability as a secondary but still monitored objective

---

## Data setup

The project uses paired `32×32×32` CT-dose cubes.

### Pilot-scale setup
Earlier pilot experiments used smaller subsets such as:
- 64 training samples
- 32 test samples

These were mainly used for:
- feasibility checks
- early model understanding
- initial pipeline debugging

### Current development setup
The larger-scale development run uses:
- **2000 training samples**
- **500 validation samples**
- hold-out test set reserved for later formal evaluation

The hold-out test set remains untouched during the current development stage.

More details are documented in [`data/README.md`](data/README.md).

---

## Phase 1 — Pilot experiments

### Main purpose
- build initial 2D and 3D pipelines
- verify that the task is learnable
- establish first regression and flow baselines

### Main outcome
The project successfully confirmed that:
- the CT-to-dose task is learnable on paired cubes
- both 2D and 3D settings can reconstruct the main beam-shaped dose pattern
- flow-based models are promising enough to justify further development

---

## Phase 2 — Analysis of pilot results

### Main purpose
- analyze normalization and scaling
- check training stability
- visualize prediction vs ground truth
- study beam-based profiles
- understand flow behavior qualitatively

### Main outcome
This phase improved confidence in:
- data preprocessing
- normalization consistency
- stability of the training pipeline
- interpretation of beam-related reconstruction quality

---

## Phase 3 — Larger-scale development run

### Main purpose
This phase moved beyond pilot feasibility and focused on:
- larger train/validation runs
- continuation beyond 30 epochs
- controlled hyperparameter tuning
- validation error analysis
- WandB / Optuna support

### Data split
- **6 training cases**
- **2 validation cases**
- **2 held-out test cases**

At the sample level:
- **2000 training samples**
- **500 validation samples**

### Best tuned baseline configuration
- `lr = 5e-4`
- `base_ch = 24`
- `batch_size = 2`
- continuation to `50` epochs

### Best phase-3 baseline result
- **best epoch:** `43`
- **best validation loss:** `1.5e-05`
- **overall validation MAE:** `0.002662`

### Main interpretation
The phase-3 tuned model is no longer only a proof of concept.  
It is a stable, well-performing development-stage baseline and remains the strongest conservative reference model for later phase-4 analysis.

### Example phase-3 figures

#### Typical validation cross-section
![Typical Sample Cross Section](docs/figures/typical_sample264_cross_section.png)

#### Typical validation along-beam profile
![Typical Along-Beam Profile](docs/figures/typical_sample264_along_beam_profile_with_pct_error.png)

#### Typical validation perpendicular profile
![Typical Perpendicular Profile](docs/figures/typical_sample264_perpendicular_profile_with_pct_error.png)

#### Worst-case cross-section
![Worst Cross Section](docs/figures/worst_sample50_cross_section.png)

#### Worst-case perpendicular profile
![Worst Perpendicular Profile](docs/figures/worst_sample50_perpendicular_profile_with_pct_error.png)

#### Average validation along-beam profile
![Average Along-Beam Profile](docs/figures/average_along_beam_profile_with_pct_error.png)

#### Average validation perpendicular profile
![Average Perpendicular Profile](docs/figures/average_perpendicular_profile_with_pct_error.png)

---

## Phase 4 — Targeted evaluation and refinement

Phase 4 shifted the focus from “general development progress” to **targeted high-dose evaluation and model behavior analysis**.

This phase includes:

- targeted evaluation with structure-sensitive metrics
- mixed-loss baseline comparison
- bone-in-beam subset analysis
- representative-case analysis
- model-specific failure-mode analysis
- routing analysis
- exploratory negative results
- a final positive refinement result

---

## Phase 4.1 — Targeted evaluation of tuned and mixed baselines

### Models compared
- **Tuned**
- **Mixed T=0.10, alpha=0.30, max_weight=3.0**
- **Mixed T=0.15, alpha=0.30, max_weight=3.0**

### Main conclusion
- **Tuned** remains the strongest conservative global / outside-region baseline
- **Mixed T=0.15** is the strongest original balanced mixed baseline
- **Mixed T=0.10** is more hard-case- / bone-in-beam-oriented

### Key phase-4 baseline summary


=======
[Phase 4 Targeted Summary](analysis/phase4/ct2dose_targeted_summary_evalT0p1.csv)


---

## Phase 4.2 — Bone-in-beam subset analysis

### Main purpose
To determine whether different models behave differently on:
- bone-in-beam candidate cases
- non-bone cases

### Main conclusion
The mixed models are not redundant:
- **Mixed T=0.10** is more promising on hard / bone-in-beam-like cases
- **Mixed T=0.15** is more balanced and stronger on the broader general set

### Bone subset summary

=======
[Bone Subset Summary](analysis/phase4/ct2dose_bone_subset_summary_evalT0p1.csv)


---

## Phase 4.3 — Representative-case analysis

### Two complementary strategies were used

#### 1. Fixed-anchor representative cases
A fair comparison on the same validation cases:
- best
- typical
- worst

This is useful for direct side-by-side model comparison.

#### 2. Model-specific representative cases
Each model was also allowed to define its own:
- best
- typical
- worst

This revealed that the models do **not** share the same failure cases.

### Main conclusion
Different models have genuinely different failure modes and should not be interpreted as simply stronger or weaker versions of each other.

---

## Phase 4.4 — Case routing analysis

### Main question
If different models specialize in different case types, can a case-dependent model-selection strategy improve performance?

### Main findings
- different models do not share the same best / worst cases
- oracle routing shows a clear upper-bound improvement
- therefore, case-dependent model selection is meaningful **in principle**

However:
- the current simple CT-only hand-crafted routing rule is still too weak
- it does not outperform the strongest single-model baselines yet

### Interpretation
Routing is a **promising direction**, but the current rule-based router should be treated as an exploratory analysis result, not yet as a final method.

### Oracle routing upper bound
[Oracle Routing Summary](analysis/phase4/ct2dose_case_routing_oracle_summary.csv)

### Routed system vs baselines



### Routed system vs baselines
[Routed vs Baselines](analysis/phase4/ct2dose_case_routing_routed_vs_baselines.csv)


---

## Phase 4.5 — Negative exploratory extensions

Two coarse biasing strategies were tested and both produced negative results.

### A. Bone-aware oversampling
This strategy increased the sampling probability of bone-in-beam candidate training cases.

#### Result
It did **not** improve the mixed baseline and degraded the main metrics.


=======
[Bone-aware Negative Result](analysis/phase4/ct2dose_bone_aware_targeted_subset_summary.csv)


### B. Beam-aware spatial weighting
This strategy emphasized voxels near the inferred beam axis.

#### Result
It also did **not** improve the mixed baseline and degraded the main structure-sensitive metrics.


=======
[Beam-aware Negative Result](analysis/phase4/ct2dose_beam_aware_targeted_summary.csv)


### Main interpretation of the negative results
These experiments suggest that:
- coarse case-level biasing is too rough
- coarse beam-axis weighting is also too rough

This motivated a more local and targeted refinement strategy.

---

## Phase 4.6 — Positive result: shoulder-aware fine-tuning

### Main idea
Instead of applying a coarse bias to full cases or full beam axes, a light refinement term was added to emphasize the **transition / shoulder region**.

The refined model is:

- **Shoulder-aware T=0.15, beta=0.10**

### Main result
This is the first clearly positive extension beyond the original mixed baselines.

Compared with the original mixed T=0.15 baseline, it improves:

- **weighted_mse**
- **high_mae**
- **peak_core_mae**
- **peak_shoulder_mae**

and these improvements hold in:
- the full validation set
- the bone-in-beam subset
- the non-bone subset

### Shoulder-aware targeted summary

[Shoulder-aware Summary](analysis/phase4/ct2dose_shoulder_aware_targeted_summary.csv)


---

## Final phase-4 conclusion

The current phase-4 picture is stable:

### Conservative baseline
- **Tuned** remains the strongest conservative global / outside-region baseline

### Best original mixed baseline
- **Mixed T=0.15** is the strongest original balanced mixed baseline

### Hard-case-oriented reference
- **Mixed T=0.10** remains a useful hard-case- / bone-in-beam-oriented reference

### Strongest current high-dose / structure-sensitive candidate
- **Shoulder-aware T=0.15** is now the strongest current candidate when the evaluation priority is shifted toward clinically relevant high-dose structure

### Negative results
- bone-aware oversampling: negative
- beam-aware spatial weighting: negative

### Routing
- promising in principle
- current CT-only rule still too weak

### Final comparison table

=======
[Final Comparison Table](analysis/phase4/ct2dose_final_comparison_table_card30.csv)


---

## Recommended reading order

For a quick but meaningful reading path:

1. Phase-3 tuned baseline
2. Phase-4 targeted evaluation of tuned vs mixed baselines
3. Bone-in-beam subset analysis
4. Representative-case analysis
5. Routing analysis
6. Negative exploratory results
7. Shoulder-aware fine-tuning and targeted evaluation

---

## Repository structure

```text
ct2dose-project/
├── README.md
├── docs/
│   └── project_overview.md
├── data/
│   ├── README.md
│   └── splits/
├── notebooks/
│   ├── 2d/
│   ├── 3d/
│   ├── phase4_baselines/
│   ├── phase4_case_analysis/
│   ├── phase4_negative_results/
│   └── phase4_positive_result/
├── analysis/
│   ├── phase3/
│   ├── phase3_error_analysis/
│   ├── phase3_wandb_optuna/
│   └── phase4/
├── docs/
│   └── figures/
├── scripts/
├── outputs/
└── raw_data/