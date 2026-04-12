# Discussion Points for April 16 Meeting

## 1. Development-stage interpretation
The current 2000/500 train/validation result, together with continuation to 50 epochs and controlled tuning, should now be regarded as a strong development-stage result.

## 2. Continuation
The continuation beyond 30 epochs produced meaningful improvement, which suggests that the earlier setup had not yet saturated.

## 3. Hyperparameter tuning
Controlled tuning further improved the result. The current best tuned configuration is:
- `lr = 5e-4`
- `base_ch = 24`
- `batch_size = 2`
- continuation to `50` epochs

## 4. Error level
The tuned model learns the dose distribution very well overall, but the error is still not consistently below 1% for the full problem.

## 5. Typical and difficult cases
Typical cases are now clearly improved, and difficult cases are also improved compared with the earlier result. The main remaining difficulty is in harder perpendicular cross-sectional structure.

## 6. Hold-out test strategy
Should the final hold-out test set remain fully untouched until the next formal stage?

## 7. Future formal comparison
For the later formal comparison, should the main target be:
- flow vs diffusion model
rather than direct regression vs flow?

## 8. Next practical step
Should the next stage focus on:
- preserving the current tuned development-stage result,
- moving later to hold-out evaluation,
- scaling further if needed,
- or preparing a diffusion-based baseline?