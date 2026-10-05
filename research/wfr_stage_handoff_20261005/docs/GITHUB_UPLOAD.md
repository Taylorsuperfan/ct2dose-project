# Upload the completed stage to the existing GitHub repository

Run these commands in the Mac terminal, not Colab. Read them one block at a time.
This adds an isolated stage folder. It does not replace the checkout or modify
older experiments. Do not commit the delivery ZIP or the whole Google Drive.

## 1. Unpack and verify outside the repository

Download `wfr_stage_handoff_20261005.zip` to Downloads and unpack it there.
The outer folder must contain `repo/`, `tools/` and `START_HERE.md`.

```bash
BUNDLE="$HOME/Downloads/wfr_stage_handoff_20261005"
python3 "$BUNDLE/repo/research/wfr_stage_handoff_20261005/scripts/check_release.py"
```

An upload-only check needs Python's standard library, not CUDA or a new model run.
Do not rerun the R0--R10 training notebook merely to upload this package.

## 2. Enter the real checkout

```bash
REPO="$HOME/Documents/pr-hesser/ct2dose-project"
cd "$REPO" &&
git rev-parse --show-toplevel &&
git status -sb &&
git --no-pager diff --cached --name-only &&
git remote -v
```

The empty/manual `Documents/master_thesis/ct2dose-project` copy is not the known
Git checkout. Do not run `git init` there to work around a directory mistake.
In the recorded setup, `github` is GitHub and `origin` is the institutional GitLab.
Confirm the current push URL; do not overwrite remotes based on this document.

```bash
git remote get-url --push github
git --no-pager log -1 --oneline
```

## 3. Make a dedicated branch from the reviewed current history

If there are unrelated changes, stop and preserve them first. Do not use reset
or clean commands. This block does not change branches when the checkout is dirty.

```bash
BRANCH="thesis/wfr-stage-handoff-20261005"
if [ -n "$(git status --porcelain)" ]; then
    printf 'STOP: review and preserve existing work before switching branches.\n'
else
    git fetch github &&
    if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git switch "$BRANCH"
    elif git show-ref --verify --quiet "refs/remotes/github/$BRANCH"; then
        git switch --track -c "$BRANCH" "github/$BRANCH"
    else
        git switch --no-track -c "$BRANCH"
    fi
fi
```

Check the printed branch before continuing. A dedicated branch still inherits
its ancestor history. Review existing history/permissions if it contains private
data. The new stage's .gitignore does not untrack old committed files.

## 4. Dry run, then import

```bash
test "$(git branch --show-current)" = "$BRANCH" &&
python3 "$BUNDLE/tools/install_into_repo.py" --repo "$REPO"
```

Review the listed destination and files. The first call changes nothing.
Then explicitly apply:

```bash
test "$(git branch --show-current)" = "$BRANCH" &&
python3 "$BUNDLE/tools/install_into_repo.py" --repo "$REPO" --apply
```

The destination is `research/wfr_stage_handoff_20261005/`. Existing identical
files are retained. Any conflicting destination content stops the operation;
there is no force-overwrite option. No Git staging or publishing is performed.

## 5. Check the exact stage and index contents

```bash
TARGET="research/wfr_stage_handoff_20261005"
python3 "$TARGET/scripts/check_release.py" &&
git add -- "$TARGET" &&
python3 "$TARGET/scripts/check_release.py" --staged --repo "$REPO" &&
git diff --cached --check &&
git --no-pager diff --cached --stat &&
git --no-pager diff --cached --name-only
```

The staged check rejects unrelated staged paths and compares actual index bytes,
not only the working tree. It is not a full privacy, institutional-permission,
intellectual-property or ancestral-history audit. Read the report, figures,
notebooks and actual file list. Obtain approval to share aggregate research
results. If Git reports nothing to commit, inspect history; do not manufacture
a change just to create another commit.

## 6. Commit and push only after review

```bash
git commit -m "Archive WFR coefficient refinement study and source" &&
git push -u github "$BRANCH"
```

Use the configured GitHub credential helper or authentication flow. Do not put a
password or token inside a source file, shell command, repository URL or chat.
If push is rejected, inspect the error and remote history. Do not use force-push.

## 7. Verify the remote branch

```bash
LOCAL=$(git rev-parse HEAD) &&
REMOTE=$(git ls-remote github "refs/heads/$BRANCH" | awk '{print $1}') &&
printf 'Local:  %s\nRemote: %s\n' "$LOCAL" "$REMOTE" &&
test -n "$REMOTE" && test "$LOCAL" = "$REMOTE" &&
printf 'PASS: the remote work branch points to this commit.\n'
```

Open the repository in GitHub and select `thesis/wfr-stage-handoff-20261005` in
the branch menu, then open `research/wfr_stage_handoff_20261005/README.md`.
The `main` page does not change simply because a work branch was pushed.
To put this work into main, create a pull request with the intended base branch
and this work branch as compare. Review all ancestry changes shown by the PR,
not just the new stage folder. Do not merge blindly if the branch contains other
unreviewed work.

## Primary references

Git behavior: https://git-scm.com/docs/git-push and https://git-scm.com/docs/git-switch
Ignore rules: https://git-scm.com/docs/gitignore
Pull requests: https://docs.github.com/en/pull-requests/how-tos/create-pull-requests/creating-a-pull-request
