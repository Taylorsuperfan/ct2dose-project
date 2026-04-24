# Trello Cards 21–31

This file contains the final Trello card plan for the current CT-to-dose project stage.

The card order reflects the intended meeting story:

1. establish the evaluation framework  
2. show case-type and model-behavior differences  
3. motivate routing  
4. record negative exploratory extensions  
5. present the two main positive outcomes  
6. end with the final project summary  

---

## Card 21
### Title
**Targeted evaluation of tuned and mixed-loss baselines**

### Description
I performed a unified targeted evaluation of the main CT-to-dose baseline models under the same evaluation protocol:

- tuned baseline
- mixed T=0.10, alpha=0.30, max_weight=3.0
- mixed T=0.15, alpha=0.30, max_weight=3.0

The evaluation focuses on:
- overall error
- weighted high-dose error
- high-dose MAE
- outside-region error
- peak-core accuracy
- peak-shoulder accuracy

Main conclusion:
the tuned model remains the strongest conservative overall baseline, while the mixed-loss direction is meaningful. Among the two mixed candidates, the T=0.15 version is the more balanced general mixed baseline, whereas the T=0.10 version remains more aggressive and more hard-case-oriented.

### Comment
Do you think this targeted evaluation setup already reflects the right priority, namely focusing on high-dose and structure-sensitive regions while still monitoring outside-region behavior, or should this priority be formalized even more explicitly?

### Suggested attachment(s)
- phase-4 targeted summary table
- corresponding CSV screenshot

### Role in the story
This card establishes the phase-4 evaluation framework and defines what counts as a meaningful improvement.

---

## Card 22
### Title
**Bone-in-beam subset analysis**

### Description
I extended the evaluation by separating the validation set into:

- bone-in-beam candidate cases
- non-bone candidate cases

Main conclusion:
the mixed models are not redundant. The T=0.10 mixed model appears more promising on harder or more bone-like cases, while the T=0.15 model is stronger as the more balanced general baseline.

### Comment
So far I used the bone-in-beam versus non-bone split as a first physically interpretable case taxonomy. Do you think this is already a useful way to characterize case difficulty, or would it be better to move toward a more general heterogeneity-based taxonomy?

### Suggested attachment(s)
- bone-in-beam subset summary table
- note with subset counts

### Role in the story
This card introduces case-type reasoning and supports the idea that different models prefer different structural conditions.

---

## Card 23
### Title
**Representative-case comparison: fixed-anchor best, typical, and worst cases**

### Description
To compare model behavior fairly on the same validation samples, I used a fixed-anchor representative-case strategy based on the tuned-model ranking and compared all models on the same best, typical, and worst cases.

Main conclusion:
the fixed-anchor comparison is useful for fair side-by-side visualization. It shows that the mixed and refined models do not behave identically to the tuned baseline and can differ substantially on hard or structured cases.

### Comment
I used a fixed-anchor representative-case strategy here for fair side-by-side comparison. For future presentations, do you think this fixed-anchor view is sufficient, or should the model-specific representative cases be emphasized more strongly?

### Suggested attachment(s)
- fixed-anchor worst-case same-scale comparison figure
- delta table for same anchor cases

### Role in the story
This card gives a fair visual comparison on the same cases before moving to model-specific behavior.

---

## Card 24
### Title
**Model-specific representative cases**

### Description
I also analyzed model-specific representative cases by allowing each model to define its own best, typical, and worst case based on its own weighted-MSE ranking.

Main observation:
the models do not share the same best and worst cases. Their model-specific representative cases are genuinely different, which confirms that the different models have different strengths, weaknesses, and failure modes.

### Comment
Since the models do not share the same best and worst cases, do you think this is already strong enough evidence that model selection should become a main direction, or would you still treat it mainly as an analysis result at this stage?

### Suggested attachment(s)
- model-specific representative-case table

### Role in the story
This card shows that the models are not just globally better or worse; they fail differently.

---

## Card 25
### Title
**Case-dependent model selection: oracle routing upper bound**

### Description
I evaluated the theoretical upper bound of case-dependent model selection using an oracle routing experiment.

Main finding:
if the best model could be selected for each case individually, the resulting aggregate performance would be substantially better than any single-model baseline. This confirms that case-dependent model selection is meaningful in principle and that there is real headroom for routing-based systems.

### Comment
The oracle result shows a clear upper-bound gain over any single model. Do you think this is already sufficient justification to continue the routing direction seriously, even though the practical router is still imperfect?

### Suggested attachment(s)
- oracle routing summary table

### Role in the story
This card justifies routing as a meaningful direction in principle.

---

## Card 26
### Title
**CT-only routing experiment**

### Description
I first tested a simple CT-only routing setup using hand-crafted and learned model-selection rules based only on CT-derived structural features.

Main conclusion:
the routing direction is meaningful, but the simple CT-only routing approaches are still too weak to outperform the strongest single-model baselines.

### Comment
The simple CT-only routing rule is still too weak. If we continue the routing direction, would you prefer a stronger learned router, or a simpler but more interpretable rule-based formulation?

### Suggested attachment(s)
- routed-vs-baselines summary
- hit-rate summary for CT-only routing

### Role in the story
This card shows the first practical routing attempts and why stronger routing features became necessary.

---

## Card 27
### Title
**Negative result: bone-aware oversampling**

### Description
I tested a bone-aware oversampling strategy on top of the mixed T=0.10 setup by increasing the sampling probability of bone-in-beam candidate training cases.

Main conclusion:
this strategy did not improve the mixed baseline. Instead, it degraded the main evaluation metrics, including weighted and high-dose metrics, and did not help even on the intended bone-in-beam subset.

### Comment
This negative result suggests that case-level oversampling is too coarse. Do you agree with that interpretation, or do you think there is still a more careful version of case-aware sampling worth trying later?

### Suggested attachment(s)
- bone-aware targeted summary
- bone-aware subset summary

### Role in the story
This is the first clear negative refinement result.

---

## Card 28
### Title
**Negative result: beam-aware spatial weighting**

### Description
I also tested a beam-aware spatial weighting strategy on top of the balanced mixed T=0.15 baseline by emphasizing voxels near the inferred beam axis.

Main conclusion:
the current beam-aware formulation did not improve the mixed T=0.15 baseline. It led to worse overall performance, worse weighted/high-dose metrics, and worse subset performance.

### Comment
The beam-aware result suggests that a simple beam-axis emphasis is still too coarse. Do you agree that this strengthens the interpretation that the transition / shoulder region is the more meaningful refinement target?

### Suggested attachment(s)
- beam-aware targeted summary
- beam-aware subset summary

### Role in the story
This negative result helps motivate why shoulder-aware refinement is a more meaningful direction.

---

## Card 29
### Title
**Positive result: shoulder-aware fine-tuning from mixed T=0.15**

### Description
Instead of introducing another coarse bias, I fine-tuned the balanced mixed T=0.15 baseline with a light shoulder-aware refinement term that explicitly emphasizes the transition / shoulder region.

Main conclusion:
this is the first clearly positive extension result beyond the original mixed baselines. The shoulder-aware T=0.15 model improves the mixed T=0.15 baseline consistently across weighted MSE, high-dose MAE, peak-core MAE, and peak-shoulder MAE. The improvement also holds in both the bone-in-beam subset and the non-bone subset.

### Comment
Do you agree with the interpretation that the transition / shoulder region is the most meaningful local target for refinement, compared with the earlier coarse case-level and beam-axis strategies?

### Suggested attachment(s)
- shoulder-aware targeted summary
- shoulder-aware subset summary
- compact summary or winner counts

### Role in the story
This is the strongest positive single-model result in the project.

---

## Card 31
### Title
**Positive result: prediction-assisted XGBoost router**

### Description
I extended the routing analysis by adding first-pass prediction features to the router input and training a learned XGBoost-based model selector on top of the existing expert models.

Expert pool:
- Tuned
- Mixed T=0.10
- Shoulder-aware T=0.15

Main conclusion:
the prediction-assisted XGBoost router is currently the strongest adaptive system in the project.

Its routing hit rate against the oracle weighted-MSE winner reaches 0.556, which is substantially better than:
- the earlier hand-crafted CT-only rule
- the CT-only learned router
- the prediction-assisted logistic-regression router
- the prediction-assisted random-forest router

In the final comparison, the prediction-assisted XGBoost router:
- matches the shoulder-aware single model on weighted MSE
- improves overall MAE
- improves outside-region MAE
- remains highly competitive on the main high-dose and structure-sensitive metrics

Interpretation:
case-dependent model selection is no longer only a theoretical idea in this project. With prediction-assisted features, routing becomes practically competitive and forms a second clear positive direction alongside the shoulder-aware single-model refinement.

### Comment
The prediction-assisted XGBoost router is now the strongest adaptive system in the project. In your view, is routing now mature enough to be considered a real method direction, rather than only an analysis result?

### Suggested attachment(s)
- router compare-vs-baselines table with XGBoost
- hit-rate summary with XGBoost
- best-router subset summary
- XGBoost feature importance figure

### Role in the story
This is the strongest adaptive-system result and the second major positive outcome of the project.

---

## Card 30
### Title
**Final model and system summary before the next meeting**

### Description
The current project picture is now stable and can be summarized at two levels:

Single-model level:
- Tuned remains the strongest conservative global / outside-region baseline.
- Shoulder-aware T=0.15 is the strongest single model when the evaluation priority is shifted toward clinically relevant high-dose and structure-sensitive regions.

Adaptive-system level:
- The prediction-assisted XGBoost router is the strongest adaptive system tested so far.
- It reaches the same weighted-MSE level as the shoulder-aware single model while improving overall MAE and outside-region behavior.

Additional conclusions:
- Mixed T=0.15 remains the strongest original balanced mixed baseline.
- Mixed T=0.10 remains a useful hard-case- / bone-oriented reference model.
- Bone-aware oversampling and beam-aware weighting were negative results.
- Routing is now no longer only a theoretical idea: the prediction-assisted router shows that case-dependent model selection can become practically competitive.

Final takeaway:
- best conservative baseline = Tuned
- best single high-dose model = Shoulder-aware T=0.15
- best adaptive system = Prediction-assisted XGBoost router

### Comment
If I continue only one main direction after this stage, would you consider the shoulder-aware single-model path or the prediction-assisted adaptive-system path more valuable?

Should the final project story emphasize the strongest single model, the strongest adaptive system, or explicitly both?

### Suggested attachment(s)
- final comparison table PNG
- final comparison CSV
- router compare-vs-baselines with XGBoost
- shoulder-aware targeted summary

### Role in the story
This is the final synthesis card and should appear last.

---

# Final Recommended Order

1. Card 21 — Targeted evaluation of tuned and mixed-loss baselines  
2. Card 22 — Bone-in-beam subset analysis  
3. Card 23 — Representative-case comparison: fixed-anchor best, typical, and worst cases  
4. Card 24 — Model-specific representative cases  
5. Card 25 — Case-dependent model selection: oracle routing upper bound  
6. Card 26 — CT-only routing experiment  
7. Card 27 — Negative result: bone-aware oversampling  
8. Card 28 — Negative result: beam-aware spatial weighting  
9. Card 29 — Positive result: shoulder-aware fine-tuning from mixed T=0.15  
10. Card 31 — Positive result: prediction-assisted XGBoost router  
11. Card 30 — Final model and system summary before the next meeting  

---

# Final Summary Sentence

The project now has three clearly separated roles:

- **Tuned** = strongest conservative baseline  
- **Shoulder-aware T=0.15** = strongest single high-dose model  
- **Prediction-assisted XGBoost router** = strongest adaptive system  