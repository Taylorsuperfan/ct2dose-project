# Discussion Points for April 16 Meeting

## 1. Development-stage interpretation
The current 2000/500 train/validation result is much more informative than the earlier pilot setup, but it is still not the final formal evaluation because the hold-out test set has not been used yet.

## 2. Generalization
So far, no strong overfitting signal is visible on the larger train/validation setup.

## 3. Hold-out test strategy
Should the final hold-out test set remain completely untouched until the next formal stage?

## 4. Data scale
Is the current 2000/500 setup already sufficient for the current development stage, or should the next step use even more training samples?

## 5. Future formal comparison
For the later formal comparison, should the main target be:
- flow vs diffusion model
rather than direct regression vs flow?

## 6. Next practical step
Should the next stage focus on:
- final hold-out testing
- larger-scale training
- or preparation of a diffusion-based baseline?