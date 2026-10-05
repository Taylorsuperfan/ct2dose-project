# Method and implementation contract

## Agreed task, not a source claim
The user accepted this controlled real-data pilot: fix the recovered Phase9G predictor B(C),
learn r=D_GT−B(C), and compare corrections against the existing Phase10D-strict system.
B is a learned old predictor, not dose-to-water. No new physical reference is inferred.
This application choice is not stated in the supervisor's HJD notes; it is separately disclosed.

The supervisor's supplied notes (pp.9–11) motivate r+=max(r,0), r−=max(−r,0),
normalization of each nonnegative spatial field, two standard spatial RFs, mass rescaling,
and an overlap product penalty. This implementation adds explicit conditional networks,
learned mass heads and endpoint auxiliary losses. It does not claim an unbalanced OT optimum
or a reaction–diffusion solver.

## Units and normalization
Use the recovered CT preprocessing and dose scaling verbatim. This historical numerical
heuristic does not establish physical units. Let r_model=D_model−B_model. Fit a SINGLE
S=sqrt(mean_train(mean_voxel(r_model²))) using selected training records only.
The new target is u=r_model/S. Conditions are CT_normalized and B_model/bmax_train.
The input scale uses training predictions only. Validation target statistics do not enter fitting.
Prediction: D_model_raw=B_model+S*u_hat; divide by the record's historical factor for stored units.
Primary scoring clamps absolute dose at zero; unclamped dose metrics remain available.
The signed residual ODE and positive/negative recombination are never clamped to nonnegative.

## Ordinary residual RF
Reuse the recovered ConditionalUNetFlow3D building blocks with in_ch=4 and base_ch=16,
fresh weights, and zero final convolution initialization. The actual concatenation is
[state, CT, scaled Phase9G, time]. Source state is zero, path u_t=t*u, target velocity u.
Optimize dense voxel FM MSE. Integrate with eight left-endpoint Euler updates.
This is NOT the original CT-initialized, midpoint-time, step-clamped RF sampler; the
frozen original upstream keeps that historical behavior unchanged.

## Spatial HJD
Take u± and M±=sum_voxel(u±). Only nonzero branches define p±=u±/M±.
The mass network predicts m±=M±/Nvox using softplus, supervised in residual-RMS units.
The output fields are Nvox*mhat±*phat±. The target's mass is NEVER supplied at inference.
Zero-target branches contribute no arbitrary FM spatial objective; mass/component losses
still supervise zero. Softplus can approach zero but does not guarantee exact zero at inference.

CT and Phase9G are CONDITIONS, not a probability interpretation of HU. Use a uniform spatial
source on [0,1]^3. Coordinates are (x,y,z) mapped to array (W,H,D). This source choice is an
explicit engineering adaptation, not the literal CT-density source in the supplied notes.
Two independent MLP velocity heads share a 3D condition encoder. Conditional pairs use
independent uniform source and target samples; linear paths are not claimed optimal-transport couplings.
Target voxel sampling uses normalized u± and within-voxel uniform jitter.

L= L_FM+ + L_FM− + 5 L_mass + L_component + L_residual + 0.1 L_overlap + 0.1 L_boundary.
Endpoint losses use fixed scrambled Sobol source particles, 4096 PER BRANCH and eight Euler
steps. Final inference uses the SAME particle count, solver, source seed and step count.
Training stochastic FM samples use1024/branch. Endpoint rollout is differentiable and therefore
this entire training objective is NOT simulation-free. The fixed finite source quadrature can
bias optimization toward its own sampling error; this is a recorded pilot limitation.

Trilinear splatting clamps out-of-domain positions to the boundary and normalizes each branch
to unit sum. Report out-of-domain fraction separately; it is NOT a discarded mass fraction.
The overlap product is sensitive to amplitude; raw branch masses and a dimensionless cancellation
ratio are also reported. Sum-of-voxel-values is not deposited energy or a physical mass without
appropriate geometry/density integration.

## Optimization and comparison
Small stage:12 selected train records, train-only monitoring,8x16updates.
Pilot:192 train records,12x32updates,batch2,AdamW lr3e-4,weight decay1e-5,seed17.
Small and pilot runs do NOT share trained weights: fresh pilot initialization isolates training budget.
New heads share record selection, batch budget, condition information and monitoring rules;
architectures/objectives/cost differ. These are controlled engineering implementations, not a
single-factor proof about HJD alone. Old Phase10D has its own historical training budget.

Pilot best selection: equal-case clipped-dose stored-unit RMSE on fixed40validation records.
Final600validation overlaps that selection subset and has been used for previous development.
Report this as development evidence. Two cases are not600patients or millions of independent observations.
The primary old profile implementation uses GT-peak diagnostic lines with legacy x=W,y=H,z=D;
it is not a target-informed prediction. Old training/feature x=D inconsistency is preserved in
old predictions, never silently corrected or rebranded as verified beam geometry.

## Persistence
Immutable source/data/config hashes, optimizer/model/RNG checkpoints, next update and history.
Save every16updates and epoch boundary. Resume verifies identity and current environment.
Exact continuation is tested only on temporary CPU examples; CUDA sampling/pooling/scatter
can remain nondeterministic. Unsaved updates may be repeated after interruptions.
A per-runtime advisory lock does not prevent writers from two different Colab VMs.
Never run simultaneous writers to the same Drive directory.

## References used for API behavior
- PyTorch, Saving and Loading Models: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch, Reproducibility: https://docs.pytorch.org/docs/stable/notes/randomness.html
These do not supply experiment results or an artificial reference field.
