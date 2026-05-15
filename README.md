
# CT-to-Dose Prediction: Regression and Flow on Paired 2D/3D Cubes

This project studies the mapping from paired CT cubes to dose cubes on a `32×32×32` dataset, using both regression-based and flow-based approaches.

The project currently includes:

- **Phase 1**: pilot experiments in 2D and 3D
- **Phase 2**: analysis and interpretation of pilot results
- **Phase 3**: a larger-scale train/validation development run for the 3D flow model, including continuation beyond 30 epochs, controlled hyperparameter tuning, and validation error analysis
- **Phase 4**: targeted evaluation, subset analysis, model-specific case analysis, routing analysis, and refinement experiments on top of the mixed-loss baselines

---

## Project goal

The goal is to learn a mapping from **CT cubes** to **dose cubes**:

`CT cube → dose cube`

and to understand how well different models reconstruct the beam-shaped dose structure, especially in **clinically relevant high-dose regions**.

In the current project stage, the main priority is **not** to optimize all regions equally.  
Instead, the focus is placed on:

1. high-dose region quality  
2. structure-sensitive region quality  
3. transition / shoulder-region accuracy  
4. global stability as a secondary but still monitored objective

---

## Current project phases

### Phase 1 — Pilot experiments
This phase focused on:
- building the 2D and 3D pipelines
- verifying that the task is learnable
- establishing initial regression and flow baselines

### Phase 2 — Analysis of pilot results
This phase focused on:
- training stability checks
- normalization and scaling analysis
- prediction / ground-truth / difference visualization
- beam-based profile analysis
- flow behavior visualization

### Phase 3 — Larger-scale development run
This phase moved beyond the pilot setup and used:
- **6 training cases**
- **2 validation cases**
- **2 fully held-out test cases**

At the sample level, the development run used:
- **2000 training samples**
- **500 validation samples**

The hold-out test set is intentionally left untouched for later formal evaluation.

### Phase 4 — Targeted evaluation and refinement
This phase focused on:
- targeted evaluation with structure-sensitive metrics
- mixed-loss comparison against the tuned baseline
- bone-in-beam subset analysis
- fixed-anchor and model-specific representative-case analysis
- case-dependent routing analysis
- exploratory biasing strategies
- a final positive refinement result based on **shoulder-aware fine-tuning**

---

## Data setup

The project uses paired `32×32×32` CT-dose cubes.

### Pilot-scale setup
Earlier pilot experiments used smaller subsets such as:
- 64 training samples
- 32 test samples

These experiments were mainly used for feasibility checks and early model understanding.

### Current development setup
The larger-scale development run uses:
- **2000 training samples**
- **500 validation samples**
- hold-out test set reserved for later use

More details are documented in `data/README.md`.

---

## Phase 3 baseline result

### Best tuned baseline configuration
- `lr = 5e-4`
- `base_ch = 24`
- `batch_size = 2`
- continuation to `50` epochs

### Best validation result
- **best epoch:** `43`
- **best validation loss:** `1.5e-05`

### Final tuned baseline evaluation
- **overall validation MAE:** `0.002662`

### Interpretation
The phase-3 tuned model is no longer only a pilot proof of concept.  
It trains stably on the current train/validation setup and serves as the strongest conservative global baseline for later phase-4 targeted evaluation.

---

## Phase 4 main results

### Main reference models
The most important phase-4 reference models are:

- **Tuned**
- **Mixed T=0.10, alpha=0.30, max_weight=3.0**
- **Mixed T=0.15, alpha=0.30, max_weight=3.0**
- **Shoulder-aware T=0.15, beta=0.10**

### Current model ranking

#### Conservative global baseline
1. **Tuned**

#### High-dose / structure-sensitive ranking
1. **Shoulder-aware T=0.15**
2. **Mixed T=0.15**
3. **Mixed T=0.10**
4. **Tuned** (still strongest outside-region baseline)

### Best current high-dose / structure-sensitive candidate
The strongest current phase-4 candidate is:

**Shoulder-aware T=0.15, beta=0.10**

Its targeted evaluation summary is:

- **overall_mae:** `0.002683`
- **weighted_mse:** `0.000028`
- **high_mae:** `0.004179`
- **outside_mae:** `0.002488`
- **peak_core_mae:** `0.036051`
- **peak_shoulder_mae:** `0.015444`

Compared with the original mixed T=0.15 baseline, it improves:
- weighted MSE
- high-dose MAE
- peak-core MAE
- peak-shoulder MAE

and the improvement holds in:
- the full validation set
- the bone-in-beam subset
- the non-bone subset

### Tuned baseline still remains important
The tuned model still remains the strongest conservative baseline for:
- global/outside-region behavior
- clean outside-region error control

Its targeted evaluation summary is:

- **overall_mae:** `0.002662`
- **weighted_mse:** `0.000038`
- **high_mae:** `0.004830`
- **outside_mae:** `0.002382`
- **peak_core_mae:** `0.047654`
- **peak_shoulder_mae:** `0.022622`

### Interpretation
The current main conclusion is therefore:

- **Tuned** remains the strongest conservative global baseline
- **Shoulder-aware T=0.15** is the strongest current model when the evaluation priority is shifted toward clinically relevant high-dose structure and transition-region quality

---

## Main phase-4 observations

### Mixed-loss baselines
The mixed-loss direction is meaningful. The two original mixed baselines already show different preferences:

- **Mixed T=0.15** is the strongest original balanced mixed baseline
- **Mixed T=0.10** is more aggressive and more hard-case- / bone-in-beam-oriented

### Bone-in-beam subset analysis
The validation set was split into:
- bone-in-beam candidate cases
- non-bone candidate cases

This analysis showed that:
- the mixed models are not redundant
- different models are better suited to different structural case types

### Representative-case analysis
Two complementary representative-case strategies were used:

1. **fixed-anchor best / typical / worst cases**  
   for fair comparison on identical validation cases

2. **model-specific best / typical / worst cases**  
   to reveal genuine differences in model-specific failure modes

The model-specific analysis showed that the models do **not** share the same worst cases.

### Routing analysis
A case-dependent routing analysis was also performed.

Main findings:
- different models do not fail on the same cases
- oracle routing shows a strong theoretical upper bound improvement
- however, the current simple CT-only hand-crafted routing rule is still too weak

So routing is:
- **promising in principle**
- but **not yet a final practical method**

---

## Negative results

Two exploratory phase-4 extensions were tested and did **not** improve the mixed baselines:

### Bone-aware oversampling
This strategy increased the sampling probability of bone-in-beam candidate training cases.  
It did not improve the mixed baselines and degraded the main evaluation metrics.

### Beam-aware spatial weighting
This strategy emphasized voxels near the inferred beam axis.  
It also did not improve the mixed baselines and degraded the structure-sensitive metrics.

### Interpretation of the negative results
These two negative results suggest that:
- coarse case-level biasing is too rough
- coarse beam-axis weighting is also too rough

In contrast, the successful shoulder-aware refinement indicates that the **transition / shoulder region** is a much more meaningful refinement target.

---

## Final experimental picture

At the current stage, the experimental picture is stable:

- **Tuned**: strongest conservative global / outside-region baseline
- **Mixed T=0.15**: strongest original balanced mixed baseline
- **Mixed T=0.10**: useful hard-case-oriented reference model
- **Shoulder-aware T=0.15**: strongest current high-dose / structure-sensitive candidate
- **Bone-aware oversampling**: negative result
- **Beam-aware weighting**: negative result
- **CT-only routing rule**: promising analysis result, but not yet strong enough as a final method

---

## Key figures

### Phase 3 baseline figures
#### 1. Typical validation cross-section
![Typical Sample Cross Section](docs/figures/typical_sample264_cross_section.png)

#### 2. Typical validation along-beam profile with percentage error
![Typical Along-Beam Profile](docs/figures/typical_sample264_along_beam_profile_with_pct_error.png)

#### 3. Typical validation perpendicular profile with percentage error
![Typical Perpendicular Profile](docs/figures/typical_sample264_perpendicular_profile_with_pct_error.png)

#### 4. Worst-case cross-section
![Worst Cross Section](docs/figures/worst_sample50_cross_section.png)

#### 5. Worst-case perpendicular profile with percentage error
![Worst Perpendicular Profile](docs/figures/worst_sample50_perpendicular_profile_with_pct_error.png)

#### 6. Average validation along-beam profile
![Average Along-Beam Profile](docs/figures/average_along_beam_profile_with_pct_error.png)

#### 7. Average validation perpendicular profile
![Average Perpendicular Profile](docs/figures/average_perpendicular_profile_with_pct_error.png)

### Phase 4 suggested summary figures
Recommended phase-4 summary figures include:
- targeted summary comparison table
- bone-in-beam subset summary table
- representative worst-case same-scale comparison
- oracle routing summary table
- routed-vs-baselines comparison table
- shoulder-aware targeted summary table
- final comparison table

---

## Repository structure

```text
ct2dose-project/
├── README.md
├── .gitignore
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
│   ├── day1/
│   ├── day2/
│   ├── day3/
│   ├── day4/
│   ├── phase2/
│   ├── phase3/
│   ├── phase3_error_analysis/
│   ├── phase3_continuation/
│   ├── phase3_tuning_round1/
│   ├── phase3_best_tuned_error_analysis/
│   ├── phase3_wandb_optuna/
│   ├── ct2dose_phase4_targeted_evaluation/
│   ├── ct2dose_phase4_case_type_analysis/
│   ├── ct2dose_phase4_case_routing_analysis/
│   ├── ct2dose_phase4_bone_aware_mixed_training/
│   ├── ct2dose_phase4_bone_aware_targeted_evaluation/
│   ├── ct2dose_phase4_beam_aware_mixed_training/
│   ├── ct2dose_phase4_beam_aware_targeted_evaluation/
│   ├── ct2dose_phase4_shoulder_aware_finetuning/
│   └── ct2dose_phase4_shoulder_aware_targeted_evaluation/
├── meeting/
│   └── 2026-04-16/
    └── 2026-04-28/

├── docs/
│   └── figures/
├── scripts/
├── outputs/
└── raw_data/
## Phase9–Phase11 profile-refinement update

The later refinement work after Phase4 focuses on professor-style profile errors and the Card17 sample491 along-beam/perpendicular trade-off.

### Current best result

The current best profile-refinement result is **Phase10D falloff-aware direction refinement**.

| Method | Along-beam x | Perpendicular y |
|---|---:|---:|
| Card17 Phase4 mixed weighted reference | 2.775% | 1.615% |
| Phase9G deployable | 4.969% | 0.774% |
| Phase10D falloff-aware | 4.363% | 0.771% |

Phase10D improves the remaining along-beam error while preserving the perpendicular advantage.

### Important inference note

The corrected rectified-flow sampler starts from CT, not zeros:

- init = CT
- midpoint Euler time
- positive integration sign
- euler_steps = 10 for Phase9D-plus / Phase9G reproduction

### Recommended next step

Evaluate Phase10D using the same global/system metrics as the previous Phase4 final summary:
overall MAE, weighted MSE, high-dose MAE, outside MAE, peak-core MAE, and peak-shoulder MAE.

See:

- `docs/phase9_to_phase11_summary.md`
- `docs/next_steps.md`
- `analysis/phase9_10_11/phase10d/`
