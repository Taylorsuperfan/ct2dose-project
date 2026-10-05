# Mathematical definitions and proposed engineering decisions

## What is supported by the sources?

The meeting recommends studying unbalanced transport first, then learning a sign
field separately and combining the two. It does not prescribe this architecture,
source measure, boundary parameterization, 256-point banks, optimizer budget,
monitor metric, or a clinical application. The reported concern about HJD
optimization is not a proof that HJD is unstable or that WFR will outperform it.

Primary mathematical source: Peng et al., *WFR-FM: Simulation-Free Dynamic
Unbalanced Optimal Transport*, arXiv:2601.06810v2, sections 3--4.
https://arxiv.org/html/2601.06810v2
Author implementation studied at commit
11c7ae99746ab023f4065f4bfea74484ce651d87:
https://github.com/QiangweiPeng/WFR-FM
The weighted OET MM update is documented by POT and Chapel et al. (2021):
https://pythonot.github.io/_modules/ot/unbalanced/_mm.html

No third-party repository is redistributed or silently edited. The solver here is
an independent NumPy specialization with scalar diagnostics, not a claim of
bitwise replication of the author's CUDA/POT pipeline. Compare its fixed-iteration
output to installed POT using `scripts/check_pot_reference.py` when available.

## 1. Prediction target and units

B(C) is the saved, frozen Phase9G calibrated dose predictor. It is not water dose.
The cache stores B and target D in legacy internal model units. Use the existing
train-only scale S and conditioning scale bs; do not refit them on validation.

    u = (D - B(C)) / S
    rho_target = abs(u)
    sign_target = sign(u)
    condition = [normalized_CT, B(C) / bs]

Training subtraction follows the existing float32 learning pipeline. The preceding
codec diagnostic used float64 subtraction; its ~1e-17 reconstruction error is not
a prediction result or a promised training precision.

For numerical voxel-sum weights q=1, a full-grid target has total sum(abs(u)).
Final reconstruction is `B(C) + S * rho_prediction * soft_sign_prediction`, divided
by the saved dose_scale_factor for stored-unit reporting. Gy and physical geometry
are not inferred. Axis labels are array axes, not verified beam directions.

## 2. Source and common mass scaling

The inference source is one unit of nonnegative numerical mass at each of the
32768 voxel centers. Its total is 32768; it does not depend on the target.
Coupling construction divides BOTH full endpoint measures by the SAME constant
N=32768. Thus source total is 1 and target total is mean(abs(u)); the unequal
ratio is preserved. This is not separate per-record unit-mass normalization.

For each selected training record, two fixed, seeded banks approximate the source
by 256 uniformly selected distinct centers and the target by 256 magnitude-weighted
sampling draws. Duplicate target positions are merged while preserving their
multiplicity weights. Each bank preserves the full target total but not its exact
voxel field. These are training approximations, not inference source sampling.
The fixed small bank set can limit accuracy; record it in every comparison.

## 3. Weighted OET and semi-couplings

Within the unit cube choose delta=1. Every endpoint pair is below pi*delta, so the
implemented travelling branch applies without silently discarding cutoff pairs.

    C_ij = -2 log cos(||x_i-y_j||/(2 delta))
    min_G >= 0  <C,G> + KL(G 1 | a) + KL(G^T 1 | b)

No entropic penalty on G itself is added. The 1000-iteration NumPy MM solver has
an iterate-change stopping test and reports whether it reached that test. It
never asserts exact optimality. No array of all iterates is stored. The matrices
are at most 256 by 256 in the default configuration and solved on CPU.

With row sums r_i and column sums c_j of G:

    gamma0_ij = G_ij a_i/r_i
    gamma1_ij = G_ij b_j/c_j

These satisfy their endpoint marginal constraints even for the approximate G.
This fact alone does NOT make the coupling globally optimal. Set q_ij=gamma0_ij
because the source total is 1. Sampling q gives conditional initial mass m0=1 and
conditional final mass m1=gamma1_ij/gamma0_ij. The code tests the endpoint total.

## 4. Conditional travelling paths and training

Let theta=distance/(2 delta), b=sqrt(m1), with initial conditional mass one:

    z_re = (1-t) + t*b*cos(theta)
    z_im = t*b*sin(theta)
    m(t) = z_re^2 + z_im^2
    x(t) = x0 + 2 delta atan2(z_im,z_re) direction
    v_target = 2 delta b sin(theta) direction / m(t)
    g_target = d(log m(t))/dt

Train the spatial fields using

    mean[m(t) * (sum_components((v-v_target)^2)
                        + delta^2 * (g-g_target)^2)]

This uses the mathematical vector squared norm (sum, not coordinate mean).
The first-week author reproduction kept the author's coordinate mean. This
adapter declares its own kappa=delta^2 and is not silently claiming unchanged
loss normalization from that reproduction.

A two-resolution 3-D convolutional encoder extracts local/global features from
CT and B(C). A time/position-conditioned MLP predicts spatial motion and growth.
An independently parameterized 3-D encoder predicts grid sign logits. Sign loss
is amplitude-weighted binary cross-entropy on |u|>0.01; that threshold masks only
sign supervision, not the target magnitude or reconstruction. Sign weighting
normalizes within the selected training batch and is an engineering choice.
Both networks receive the same record batch, but have separate parameters.
There is no endpoint-ODE backpropagation or extra profile loss in this first pilot.

## 5. Inference, boundary parameterization and reconstruction

To keep particles inside the computational box without discarding them, represent
x=sigmoid(y) and learn dy/dt through an unconstrained head. The fitted spatial
velocity is v=x(1-x)*dy/dt; the same definition is used in the velocity loss.
This box-tangent parameterization is a proposed boundary choice, not an author
WFR-FM architectural feature. Integrate y and log(m) using 16 Heun steps. Growth
can be negative, but finite multiplicative growth does not create negative mass.
Extreme/nonfinite states cause a stop, not mass clamping or target-based rescaling.

Inference uses all 32768 deterministic source centers, queried in chunks of8192.
Changing a query chunk does not change source support. These are not a32768^2 OT
matrix. Trilinear deposition creates nonnegative cell weights, then a grid-valued
soft sign `tanh(logit/2)` is multiplied. Grid-sign-last differs from the previous
particle-sign-first numerical operator. Its signed net sum is not constrained to
an independently prescribed particle signed total. Soft-sign amplitude shrinkage
is reported rather than repaired with target information.

Final absolute dose is clipped at zero for the primary historical comparison;
raw dose and hard-sign sensitivity are also saved. The primary method is fixed
in advance as soft grid sign. Do not select whichever sign variant wins afterward.

## 6. Limits

The model is a numerical conditional error-correction model, not explicit photon
transport, dose-to-water prediction, signed WFR geodesic proof or clinical software.
Finite supports, finite MM iterations, boundary choices, discretization and neural
approximation limit direct use of original optimality theorems. The exact all-zero
target case stops this nonzero-target pilot; no epsilon target is invented. The
existing sampled real cases are nonzero, but other future data need a separately
specified zero-target handling mechanism.

Uniform source, width16/64, two banks, point counts, delta1, AdamW3e-4, sign weight1,
threshold0.01, Heun16 and update budgets are declared engineering choices, not
optimized values or supervisor-prescribed constants. No performance improvement
is claimed before running the actual real-data experiment.
