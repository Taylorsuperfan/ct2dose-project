# Discussion Points for April 16 Meeting

## 1. Development-stage interpretation
The current 2000/500 train/validation result, especially after continuation to 50 epochs, is much more informative than the earlier pilot setup. It should now be regarded as a strong development-stage result.

## 2. Training continuation
The controlled continuation from 30 to 50 epochs produced meaningful improvement, which suggests that the model had not yet saturated at 30 epochs.

## 3. Hold-out test strategy
Should the final hold-out test set remain fully untouched until the next formal stage?

## 4. Error level
The model now learns the dose distribution very well overall, but the error is still not consistently below 1%. Typical cases are in the few-percent range, while difficult cases remain clearly higher.

## 5. Main remaining difficulty
The main remaining difficulty seems to be in harder perpendicular cross-sectional structure rather than in the main along-beam decay pattern.

## 6. Future formal comparison
For the later formal comparison, should the main target be:
- flow vs diffusion model
rather than direct regression vs flow?

## 7. Next practical step
Should the next stage focus on:
- preserving the current result and later evaluating the hold-out test,
- scaling further,
- or preparing a diffusion-based baseline?