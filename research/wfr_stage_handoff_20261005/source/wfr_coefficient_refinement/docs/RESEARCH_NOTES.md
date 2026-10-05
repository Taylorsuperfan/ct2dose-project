# What the current results suggest, and what is still open

## Evidence from this project

The real-data parent has better whole-cube RMSE than the saved Practical final system, but worse array-x profile error. That trade-off appeared on monitor40 before validation600 was scored. It cannot be explained solely by the relative-error denominator, because absolute x-profile RMSE worsened too.

The recent monitor40 diagnostic found positive recall around 28.0% and negative recall around 88.9%. A weighted accuracy alone hides that asymmetry. The same diagnostic found an average predicted-to-target magnitude ratio around 1.187 and local magnitude undercoverage at many profile locations. A globally large magnitude can still be too small in particular voxels.

The fixed-magnitude bounded-coefficient oracle improved the reported x metric from about 9.876% to 5.338% on these same40 records; the saved Practical final system was about7.978% there. These numbers are not validation600 results. The oracle uses every target value independently, so a finite conditional network is not promised to reach it.

The frozen coefficient retained about39.54% of predicted magnitude. Hardening signs made the real-dose results worse. Soft outputs are therefore not merely a nuisance to be removed: they are also limiting uncertain or misplaced corrections.

The current parent used small, fixed approximate coupling banks and a bounded-domain velocity parameterization. These are possible sources of magnitude error, not proven causes. The point count, geometry convention and finite solver should remain visible in the thesis.

## Why the next experiment is small

We now have enough evidence to train, not to add another chain of diagnostic gates. Keeping magnitude fixed makes the first intervention easy to interpret and cheap after caching. The experiment does not change the architecture, source, particle count, coupling, integration or data split. A second arm adds a profile loss; that is the key contrast.

This is a supervised composite refinement over an existing WFR representation. It is not a fresh WFR geodesic solver. If it works, it shows that the representation was useful and that the coefficient objective mattered. It does not prove the WFR geometry was uniquely responsible for improvement.

## Theory note: a soft sign is not necessarily an ordinary class probability

At a fixed condition, let a random residual U have a finite first moment. Ignore masks, finite batches and model error for this population calculation. Amplitude-weighted binary cross-entropy is minimized by

    p_w = E[|U| 1_{U>0} | condition] / E[|U| | condition].

Hence

    2 p_w - 1 = E[U | condition] / E[|U| | condition].

If an independent magnitude estimator produced exactly E[|U||condition], their product would recover E[U|condition]. This is a useful ideal identity, not a statement that the current WFR endpoint estimator equals that conditional expectation. The actual mask and batch-dependent normalization introduce further distinctions.

It also means that simply class-balancing the BCE or temperature-sharpening the output can change the meaning of the coefficient. Standard probability calibration literature does not make this amplitude-weighted output a calibrated sign probability automatically. That is why the present study supervises the product rather than treating sign accuracy as the sole objective.

## Idea 1: attainable-set-aware correction learning

For fixed a>=0, the best pointwise coefficient in [-1,1] projects U onto [-a,a]. The unavoidable absolute error is (|U|-a)_+. This gives a useful diagnostic certificate and a way to separate two limitations: coefficient error within the envelope, and missing magnitude outside it.

A possible later study would use this envelope to route learning effort. For example, a training-only undercoverage map could supervise a separate capacity predictor or identify where magnitude refinement is necessary. Oracle maps must not be supplied at inference. The predictor would need to infer its own map from observable conditions.

One can report an excess squared error, (a*c-U)^2-(|U|-a)_+^2. Subtracting that target-dependent constant does NOT change the coefficient gradient; it is not, by itself, an optimization innovation. A genuinely different optimizer or loss would need an explicit derivation and ablation.

Novelty is not established. The contribution would be the specific certificate-guided allocation and its empirical/theoretical analysis in this setting, not the elementary projection formula.

## Idea 2: repair the magnitude in square-root coordinates

If coefficient refinement reaches an envelope limit, a controlled local magnitude adaptation could penalize displacement in sqrt(a), inspired by Fisher-Rao reaction geometry. For example, a nonnegative a_new could be supervised with a fidelity term to the true residual magnitude and a trust penalty proportional to sum((sqrt(a_new)-sqrt(a_parent))^2).

The goal is local correction rather than a single global rescaling: the current data show local shortage alongside global excess. This is not yet implemented. A multiplicative a_new=a_parent*exp(h) cannot create support where a_parent=0. A square-root parameterization (sqrt(a_parent)+d)^2 has its own zero-gradient issue at a_parent=d=0. Zero support and initialization must therefore be designed, not hidden by an unexplained epsilon.

A post-hoc Fisher-Rao-inspired penalty does not preserve the original WFR geodesic theorem. Establishing a coherent dynamic extension would require more analysis.

## Idea 3: treat profile and volume as competing objectives

The project already has a measurable trade-off. Instead of trying many hand-picked scalar weights, a later experiment could measure gradient alignment between volume and profile objectives, then compare a standard weighted sum with a multi-objective update. PCGrad or Pareto-based methods are established baselines, not new inventions.

An empirical gradient conflict would justify this direction; a bad profile metric alone does not prove the gradients conflict. Such updates offer no guarantee of clinical safety or global Pareto optimality for this finite nonconvex problem. The present release uses neither PCGrad nor a hidden adaptive weighting scheme, so the A/B comparison stays simple.

## Idea 4: model a genuinely time-dependent signed field

Writing f=s*rho with positive rho and a WFR transport field gives, by the product rule,

    partial_t f + div(f*v) = g*f + rho*(partial_t s + v dot grad(s)).

The endpoint product used today does not specify the material derivative of s. A more ambitious research direction would model it and define an explicit cost for sign changes. This could be more than an endpoint correction head, but it is a different mathematical problem.

For example, writing s=cos(theta) suggests evolving an angle rather than forcing an exactly positive initial sign through ds/dt=(1-s^2)h, which would trap s=1 under finite regular h. An angular action weighted by rho is a possible idea; it is NOT an established signed WFR metric. Periodicity, non-unique representations, zero-magnitude sets, endpoint conditions and well-posedness would all need to be addressed. No code or metric claim for this direction is included here.

## Practical priority

Run the two current refinement arms first. If both improve substantially, prioritize replication and whether the profile arm adds value. If neither learns even on the training records, inspect optimization or coefficient capacity before starting a new theory project. If the coefficient improves but profile undercoverage remains, magnitude repair becomes the next bounded intervention. A full signed dynamic theory is a longer research direction, not a prerequisite for the next useful experiment.

The recorded meeting asks for a thoughtful choice and a thorough analysis of one main route. It does not guarantee that the preferred route is more stable or more accurate. Keep that distinction in the thesis.

## Primary reading

1. Peng et al. WFR-FM: Simulation-Free Dynamic Unbalanced Optimal Transport. arXiv:2601.06810v2, 2026. https://arxiv.org/html/2601.06810v2
2. Chizat, Peyre, Schmitzer and Vialard. Unbalanced Optimal Transport: Dynamic and Kantorovich Formulation. arXiv:1508.05216; Journal of Functional Analysis, 2018. https://arxiv.org/abs/1508.05216
3. Sener and Koltun. Multi-Task Learning as Multi-Objective Optimization. NeurIPS2018. https://arxiv.org/abs/1810.04650
4. Yu et al. Gradient Surgery for Multi-Task Learning. NeurIPS2020. https://arxiv.org/abs/2001.06782
5. Guo et al. On Calibration of Modern Neural Networks. ICML2017. https://proceedings.mlr.press/v70/guo17a.html

These references motivate background and possible directions. They do not establish novelty or efficacy for the current CT-to-dose extension. No exhaustive novelty search has been completed.
