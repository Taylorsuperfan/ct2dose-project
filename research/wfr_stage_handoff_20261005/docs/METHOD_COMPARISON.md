# Why these methods are related, and where they differ

## Separate three comparisons

The latest study compares a composition-supervised coefficient head with the
same head trained with an added profile loss. A broader thesis comparison asks
why a WFR magnitude-plus-coefficient approach was pursued rather than two
positive/negative flows. The final Practical system is a third comparison: an
established dose predictor with purpose-built downstream refinement. These are
not the same question and should not share a vague label such as "the old model."

## Direct residual rectified flow

The saved direct baseline treats the complete residual volume as a real-valued
state. Its source is zero, the path is z_t=t u, and its target state-space velocity
is u. A conditioned 3-D network learns this velocity and is integrated using the
saved eight-step Euler definition. Real-valued coordinates can be negative; the
method does not interpret each residual voxel value as nonnegative spatial mass.

This distinction is important: conservation of probability in a generative
state-space flow does not say that every coordinate is nonnegative or that the
sum of image intensities is conserved. Therefore WFR is not mathematically
required merely to obtain negative residual values. The project uses it to study
an explicit nonnegative spatial measure with changing total weight.

Advantages are a direct signed target, a compact conceptual pipeline and no
extra amplitude/sign representation. Its limitations are weaker explicit
measure-level transport structure and the observed volume/profile trade-off.
The saved method contract is `source/phase9g_signed_real_v1/docs/METHOD.md`.
For the underlying rectified-flow formulation, see Liu et al., arXiv:2209.03003.

## Two nonnegative spatial flows with Hahn--Jordan targets

For a signed target u, define u+ = max(u,0) and u- = max(-u,0), so u=u+-u-.
The canonical target components are disjoint. Each nonzero branch is normalized
by its own voxel-sum M+ or M-, and a spatial rectified flow learns its normalized
shape. Conditional mass heads predict the two total weights at inference; GT
masses are not given to the prediction function. The output is reconstructed
as predicted positive field minus predicted negative field.

The actual previous implementation shares a 3-D condition encoder between two
velocity heads, uses a uniform spatial source rather than treating raw HU as a
probability density, and predicts branch totals with softplus. Its loss includes
flow, branch-mass, component, final-residual, overlap and boundary terms. It uses
4,096 particles per branch and eight Euler steps. Endpoint auxiliary terms are
differentiated through a short rollout, so the entire previous objective is not
purely simulation-free even though its core RF velocity regression is.

Why penalize overlap? If only p-n is supervised, adding the same nonnegative h
to both p and n leaves their difference unchanged. Positive overlap can therefore
hide unnecessary cancellation. Component targets and an overlap penalty restrict
this freedom. The canonical mathematical decomposition itself is not ambiguous;
soft predicted outputs need not satisfy exact disjointness. A finite penalty
does not prove it. Swapping the two branches generally negates the residual, not
leaves it unchanged. A meeting concern about symmetry is motivation to examine
optimization, not a theorem of inevitable failure.

This route closely matches a signed correction task and reuses familiar spatial
RF components. Its liabilities include two shapes/totals, branch overlap and
cancellation, special zero-branch handling, and particle/reconstruction cost.
These are design concerns, not proof that it is unstable or worse: its saved
x-profile result remains better than the current WFR refinement.

## WFR nonnegative magnitude, then a bounded signed coefficient

WFR augments the spatial continuity equation with a reaction term:

\[
\partial_t\rho+\nabla\cdot(\rho v)=g\rho,
\qquad \dot x=v(t,x),\quad \dot m=g(t,x)m.
\]

For positive initial weights and finite integrated growth,

\[
m(t)=m(0)\exp\left(\int_0^t g(\tau,x_\tau)\,d\tau\right)>0.
\]

Negative growth reduces nonnegative weight; it does not create negative weight.
WFR geometry weighs transport motion and reaction, for example through an action
proportional to integral rho (|v|^2 + delta^2 g^2). WFR-FM learns velocity and
growth targets along analytically constructed conditional paths. Its claims
about optimal paths depend on the appropriate coupling and loss assumptions.
See Peng et al., arXiv:2601.06810v2, sections 3--4.

Our conditional adapter uses numerical |u| voxel weights, finite weighted UOT
banks, CT and B(C) conditioning, a box-preserving position parameterization,
log-mass integration and trilinear grid reconstruction. The coupling and neural
approximations do not establish an exact full-grid optimum. The source is a
numerical uniform grid, not water dose and not a positive interpretation of HU.
The common training normalization preserves total-weight ratios rather than
normalizing every record to mass one.

A separate coefficient c=tanh(logit/2) makes the endpoint signed: u_hat=a*c.
The latest stage keeps a fixed and improves c by supervising u_hat and its
array-x profile. The coefficient may compensate overestimated a by choosing
|c|<1. It cannot compensate undercoverage |u|>a under the bounded parameterization.
The ideal pointwise coefficient clip(u/a,-1,1) is an oracle, not a trained network.

Advantages are explicit displacement and growth, one nonnegative magnitude flow
rather than two subtracting spatial branches, and a decomposable pipeline in
which magnitude, sign and numerical reconstruction can be studied separately.
Costs include UOT approximation, an additional growth field, positivity/zero-mass
issues, bounded-coefficient shrinkage, spatial undercoverage and inference
quadrature. One spatial flow does not imply lower actual compute than the saved
HJD implementation: this parent uses 32,768 source particles and 16 Heun steps.

It fits the supervisor's recommendation to study unbalanced transport, learn the
sign problem separately and then combine them. It does not establish superior
training stability merely because this route was preferred. The endpoint
product is not itself a proof of a signed WFR geodesic. The displayed PDE is
transport--reaction (advection--reaction): no Laplacian diffusion term is included.
Its artificial flow time and numerical weights are not photon travel time or a
validated energy-deposition model. Conditional refinement of B(C)'s error is not
the planned masked-water-reference application.

## The final Practical system

The comparator is the recovered Phase10D-strict pipeline, not an early Phase3
checkpoint and not a new approximate reconstruction. It builds on the old
rectified-flow prediction and calibrated Phase9G chain, then applies a small
bounded refinement using image/prediction/geometry features and direction,
profile, log-profile, falloff and protection-oriented training terms. Its
historical array conventions are retained rather than silently relabeled.

It is the strongest principal x-profile comparator in this reported table,
and its loss design directly addresses that objective. It also has a longer,
specialized development history and upstream exposure limitations. The word
"strict" describes the refinement-head split repair; it does not by itself
prove that every upstream component had never seen an evaluation case.

## What the current numbers establish

The latest A/B study supports adding profile supervision under the fixed
magnitude and selection setup. The profile-selected system improves the listed
global and x measures over the saved direct residual flow. Relative to the
Hahn--Jordan and Practical comparators it reduces global errors but does not
surpass the main x-profile metric. WFR geometry alone has not been isolated as
the cause. It remains a promising research direction with a demonstrated useful
refinement, not a universally better or physically validated solver.
