# Meeting Summary

## Project
CT-to-Dose Prediction on Paired 3D CT/Dose Cubes

## Meeting purpose
This meeting is intended to summarize the current project status, clarify the strongest current results, and discuss which direction should be treated as the main continuation path.

---

## 1. High-level project status

The project has progressed through four main stages:

### Phase 1
Pilot 2D and 3D experiments:
- pipeline construction
- feasibility verification
- early baseline establishment

### Phase 2
Pilot-result analysis:
- normalization and scaling checks
- prediction / ground-truth / difference visualization
- along-beam and perpendicular profile analysis
- structural interpretation of beam reconstruction

### Phase 3
Larger-scale 3D development baseline:
- continuation beyond 30 epochs
- controlled hyperparameter tuning
- WandB / Optuna-assisted search
- stable tuned baseline established

### Phase 4
Targeted evaluation and refinement:
- structure-sensitive metrics
- bone-in-beam subset analysis
- representative-case analysis
- routing analysis
- negative exploratory extensions
- positive shoulder-aware refinement
- positive prediction-assisted adaptive routing

---

## 2. Current main conclusions

### Best conservative baseline
**Tuned**

Interpretation:
- strongest global / outside-region conservative baseline
- still the cleanest reference model

### Best single high-dose model
**Shoulder-aware T=0.15**

Interpretation:
- strongest single model under high-dose / structure-sensitive evaluation
- improves weighted MSE, high-dose MAE, peak-core MAE, and peak-shoulder MAE

### Best adaptive system
**Prediction-assisted XGBoost router**

Interpretation:
- strongest routing-based adaptive system so far
- matches the shoulder-aware model on weighted MSE
- improves overall MAE
- improves outside-region behavior
- remains highly competitive on structure-sensitive metrics

---

## 3. Important intermediate findings

### Mixed-loss baselines
- Mixed T=0.15 = strongest original balanced mixed baseline
- Mixed T=0.10 = useful hard-case- / bone-oriented reference

### Bone-in-beam subset analysis
- different models behave differently on different case types
- the mixed baselines are not redundant

### Representative-case analysis
- fixed-anchor cases are useful for fair side-by-side comparison
- model-specific representative cases show genuinely different failure modes

### Routing analysis
- oracle routing shows a clear upper-bound improvement
- early CT-only routing was too weak
- prediction-assisted routing is now practically competitive

---

## 4. Negative results

### Bone-aware oversampling
- did not improve the mixed baseline
- too coarse as a case-level intervention

### Beam-aware spatial weighting
- did not improve the mixed baseline
- too coarse as a beam-axis-level intervention

### Interpretation
These negative results strengthen the idea that the meaningful refinement target is not a coarse case-level or beam-axis-level bias, but a more local transition-region refinement.

---

## 5. Positive results

### Positive result 1
**Shoulder-aware fine-tuning**

Main interpretation:
- the transition / shoulder region appears to be the most meaningful local target for refinement

### Positive result 2
**Prediction-assisted XGBoost router**

Main interpretation:
- case-dependent model selection is no longer only a theoretical idea
- adaptive selection can become practically competitive when first-pass prediction features are included

---

## 6. Main discussion questions

### Question 1
If only one main continuation path is chosen after this stage, should it be:
- further shoulder-aware refinement
- or further adaptive-system development?

### Question 2
Is the transition / shoulder region now clearly the most meaningful target for refinement?

### Question 3
Should the final project story emphasize:
- the strongest single model
- the strongest adaptive system
- or explicitly both?

### Question 4
Is the current high-dose / structure-sensitive evaluation philosophy already the right long-term framing for the project?

### Question 5
Is routing now mature enough to be treated as a real method direction?

---

## 7. Results to show during the meeting

Recommended main figures / tables:
- phase-4 targeted summary
- bone-in-beam subset summary
- fixed-anchor worst-case comparison
- oracle routing summary
- shoulder-aware targeted summary
- prediction-assisted router comparison with XGBoost
- final comparison table

---

## 8. One-sentence takeaway
The project now has three clearly separated roles:
- **Tuned** = strongest conservative baseline
- **Shoulder-aware T=0.15** = strongest single high-dose model
- **Prediction-assisted XGBoost router** = strongest adaptive system

---

## 9. Optional short speaking summary
A short verbal summary for the meeting:

The project started from feasibility-oriented 2D and 3D pilot experiments, moved into a stable tuned 3D baseline, and then shifted toward targeted evaluation focused on high-dose and structure-sensitive regions. Two coarse refinement strategies failed, while shoulder-aware fine-tuning became the strongest single-model refinement. In parallel, prediction-assisted XGBoost routing became the strongest adaptive system. The main discussion point now is which of these two positive directions should be treated as the more meaningful next step.