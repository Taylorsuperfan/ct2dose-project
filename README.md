
# CT-to-Dose Prediction: Regression and Flow on Paired 2D/3D Cubes

This project studies the mapping from paired CT cubes to dose cubes on a `32×32×32` dataset, using both regression-based and flow-based approaches.

The project currently includes:

- **Phase 1**: pilot experiments in 2D and 3D
- **Phase 2**: analysis and interpretation of pilot results
- **Phase 3**: a larger-scale train/validation development run for the 3D flow model, including continuation beyond 30 epochs and additional validation error analysis

---

## Project goal

The goal is to learn a mapping

$$
\text{CT cube} \rightarrow \text{dose cube}
$$

and to understand how well flow-based models can reconstruct the beam-shaped dose structure.

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

At the sample level, the current development run uses:
- **2000 training samples**
- **500 validation samples**

The hold-out test set is intentionally left untouched for later formal evaluation.

---

## Data setup

The project uses paired `32×32×32` CT-dose cubes.

### Pilot-scale setup
Earlier pilot experiments used smaller subsets such as:
- 64 training samples
- 32 test samples

These experiments were mainly used for feasibility checks and early model understanding.

### Current development setup
The current larger-scale flow run uses:
- **2000 training samples**
- **500 validation samples**
- hold-out test set reserved for later use

More details are documented in `data/README.md`.

---

## Current best phase-3 flow result

### Configuration
- conditional 3D U-Net flow
- `base_ch = 24`
- `batch_size = 2`
- `lr = 3e-4`
- continued from 30 to 50 epochs

### Best validation result
- **best epoch:** `49`
- **best validation loss:** `1.7157e-05`

### Final epoch result
- **final train loss:** `3.7817e-05`
- **final validation loss:** `2.3082e-05`
- **final validation MSE:** `1.9592e-05`
- **final validation MAE:** `0.002913`

---

## Main observations

### Training behavior
- train and validation losses decrease strongly overall
- no strong persistent overfitting signal is visible on the current `2000 / 500` setup
- continuation from 30 to 50 epochs gives meaningful additional improvement

### Validation reconstruction
Using the improved 50-epoch checkpoint, validation examples show that:
- the main beam-shaped dose structure is reconstructed very well
- the beam entrance region is localized correctly
- the along-beam decay pattern is captured especially well
- the perpendicular structure is also reconstructed well for typical cases

### Validation error analysis
- the best validation samples are now around **`3.46% – 3.61%`** global relative error
- a **typical case** shows profile percentage errors of about:
  - **`2.67%`** along the beam
  - **`3.76%`** perpendicular to the beam
- a **worst case** still shows clearly larger error, especially in the perpendicular direction:
  - global relative error: **`16.64%`**
  - along-beam mean percentage error: **`3.90%`**
  - perpendicular mean percentage error: **`12.13%`**

### Interpretation
The current flow model is no longer only a pilot proof of concept.  
On the larger train/validation setup, it trains stably, improves beyond 30 epochs, and reconstructs the main beam-related dose structure very well.

At the same time, the error is still not consistently below 1%, so the current result should be understood as a **strong development-stage result** rather than the final formal evaluation.

---

## Key figures

### 1. Phase-3 continuation: scaled train vs validation loss
![Phase 3 Continuation Loss Scaled](docs/figures/phase3_continuation_loss_scaled_0_to_5e-4.png)

### 2. Typical validation cross-section
![Typical Sample Cross Section](docs/figures/typical_sample436_cross_section.png)

### 3. Typical validation along-beam profile with percentage error
![Typical Along-Beam Profile](docs/figures/typical_sample436_along_beam_profile_with_pct_error.png)

### 4. Typical validation perpendicular profile with percentage error
![Typical Perpendicular Profile](docs/figures/typical_sample436_perpendicular_profile_with_pct_error.png)

### 5. Worst-case perpendicular profile with percentage error
![Worst Perpendicular Profile](docs/figures/worst_sample50_perpendicular_profile_with_pct_error.png)

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
│   └── 3d/
├── analysis/
│   ├── day1/
│   ├── day2/
│   ├── day3/
│   ├── day4/
│   ├── phase2/
│   ├── phase3/
│   └── phase3_error_analysis/
├── meeting/
│   └── 2026-04-16/
├── docs/
│   └── figures/
├── scripts/
├── outputs/
└── raw_data/