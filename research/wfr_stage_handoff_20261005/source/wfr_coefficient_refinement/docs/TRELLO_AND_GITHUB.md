# Recording and sharing the work

No Trello card or GitHub repository has been changed by this package. These are draft records and local instructions.

## Doing

**Title: Train bounded dose-correction coefficients with a frozen transport model**

The existing Wasserstein–Fisher–Rao magnitude predictor is held fixed. Two refinements start from the same coefficient weights and use the same training records, batches and extra update budget. One learns the actual signed correction; the other adds array-x profile supervision. A predeclared monitor rule protects volume and lateral metrics. If no candidate qualifies, the previous prediction is retained and the failed last-state result is still reported.

Completion requires both384-update histories, fixed selections and clearly labelled additional compute. Oracle figures are motivation, not performance results.

## To do

**Title: Compare the two refinements with the previous dose-prediction systems**

After locking both selections, score the same600 development cubes using the existing metrics. Compare the parent WFR, both new selected states, Phase10D-strict, the previous positive/negative method, ordinary signed residual flow, and the shared calibrated predictor. Include each case separately, the fixed-budget last states and the extra training cost. This is not a blind clinical test.

**Title: Decide whether the remaining limit is coefficient learning or magnitude coverage**

Use the controlled A/B result to choose one follow-up. Candidate directions include attainable-set-aware learning, local square-root magnitude repair, and multi-objective gradient methods. These are research ideas, not completed features or confirmed novel contributions.

## Done, after the corresponding work actually finishes

**Title: Complete the fixed-magnitude oracle review**

The review used saved monitor40 factors only. It identified low positive recall and substantial bounded-coefficient headroom, with local magnitude shortages still present. It did not train a model or create a new validation result. Attach the existing summary, not patient arrays.

When the new A/B experiment finishes, move its Doing card to Done only with its actual outcome. "No eligible refinement" is a valid completed experiment, not a reason to claim improvement.

## Files to publish

The source package can be placed under `research/wfr_coefficient_refinement/`. Keep its unchanged `parent_source/cwfr` copy with the source identity record. Include the notebook, configuration, protocol, readable notes, tests and verification report.

Do not publish `frozen.local`, `runs.local`, `evaluations.local`, raw arrays, weights, `.private` records, patient/sample maps, runtime logs with private paths, or a complete unreviewed Drive export. Aggregate report CSVs and figures still require content and publication review. A private repository is not a substitute for permission to share data.

## Local installation without overwriting another project

Use the existing repository, not the similarly named manual copy:

```bash
cd "/Users/ruicongrong/Documents/pr-hesser/ct2dose-project"
git status -sb
git remote -v
```

Review any existing work before creating a branch. Do not reset it.

```bash
git switch -c thesis/wfr-coefficient-refinement
python3 "$HOME/Downloads/wfr_coefficient_refinement/scripts/install_into_repo.py" \
  --repo "/Users/ruicongrong/Documents/pr-hesser/ct2dose-project"
```

The first call is a dry run. After checking its file list:

```bash
python3 "$HOME/Downloads/wfr_coefficient_refinement/scripts/install_into_repo.py" \
  --repo "/Users/ruicongrong/Documents/pr-hesser/ct2dose-project" --apply
git add -- research/wfr_coefficient_refinement
git --no-pager diff --cached --stat
git --no-pager diff --cached --name-only
git diff --cached --check
```

Review the actual staged content before committing. If suitable for publication:

```bash
git commit -m "Add controlled profile-aware coefficient refinement"
git push -u github thesis/wfr-coefficient-refinement
```

The `github` remote is the GitHub connection in the recorded setup; `origin` was GitLab. Confirm `git remote -v` before pushing. Do not use force-push to bypass a rejection. A pushed work branch does not automatically update `main`.
