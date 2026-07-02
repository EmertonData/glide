---
name: dependabot
description: Handle open dependabot PRs automatically. Finds the oldest open dependabot PR, classifies it, applies the version bump to the branch, merges main to eliminate conflicts, and posts a summary comment. Merging is left to the maintainer. Use this skill whenever the user says "handle dependabot PRs", "take care of dependabot", "process dependabot updates", "triage dependabot", "deal with dependabot", or anything similar about pending dependabot pull requests.
---

## Overview

Processes the oldest open dependabot PR and makes it ready for maintainer review. Applies the dependency bump to the branch first, then merges the latest main to ensure the branch is conflict-free.

**Dependency types:**
- **Direct**: declared in `pyproject.toml` — bump the version constraint and regenerate the lock file.
- **Indirect-python**: transitive dep, not in `pyproject.toml` — regenerate the lock file only.
- **GitHub Actions**: bump the action version in workflow files.

Merging is left to the maintainer.

---

## Workflow

### Step 1 — Discover open dependabot PRs

```bash
gh pr list --author app/dependabot --state open --json number,title,headRefName,url --jq 'sort_by(.number)'
```

Stop and report if no PRs are found. Select the one with the **lowest PR number** as the target.

### Step 2 — Classify the PR

Extract the package name (token after "Bump", lowercased). Search `pyproject.toml` (`[project].dependencies` and all `[dependency-groups]` keys) for that name (case-insensitive):

- **direct**: found — note the group (`project`, `dev`, `doc`, etc.) and current constraint (e.g. `>=0.0.53`).
- **indirect-python**: Python package, not found in `pyproject.toml`.
- **github-actions**: owner/action pattern (e.g. `actions/checkout`).

For a **github-actions** PR, confirm the version format as it appears in the workflow files:

```bash
grep -r "<action>" .github/workflows/
```

Print the classification and target version, then ask: *"Ready to process this PR?"* Wait for confirmation before touching any branch.

### Step 3 — Process the PR branch

#### 3a. Check out the branch

```bash
git fetch origin
git checkout <headRefName>
```

#### 3b. Apply the version bump

**Direct PR:** run `uv add` for this entry, then regenerate the lock file:

- `[project].dependencies` → `uv add "<package>>=<target-version>"`
- `[dependency-groups] dev` → `uv add --group dev "<package>>=<target-version>"`
- `[dependency-groups] doc` → `uv add --group doc "<package>>=<target-version>"`

Preserve the existing constraint operator.

```bash
make venv
```

**Indirect-python PR:** no `pyproject.toml` change. Regenerate the lock file only:

```bash
make venv
```

**GitHub Actions PR:** replace the old version with the new across all workflow files:

```bash
perl -pi -e 's|<action>@<old-version>|<action>@<new-version>|g' .github/workflows/*.yml
```

If `make venv` fails, check out `main`, record the failure, notify the user, and stop.

#### 3c. Commit the version bump

```bash
git add pyproject.toml uv.lock .github/workflows/
git commit -m "chore: apply dependabot update"
```

If `git commit` says "nothing to commit", note it and continue to 3d.

#### 3d. Merge main

```bash
git merge origin/main
```

If the merge is clean (no conflicts) or already up-to-date, skip to 3f.

If there are conflicts, resolve each type as follows:

**`pyproject.toml` conflicts:** for each conflict block, compare the version strings from both sides and keep the higher version. Remove all conflict markers.

```bash
git add pyproject.toml
```

**`uv.lock` conflicts:** always resolve by regenerating:

```bash
make venv
git add uv.lock
```

**`.github/workflows/` conflicts:** for each conflict block, keep the higher version string. Remove all conflict markers.

```bash
git add .github/workflows/
```

If `make venv` fails during conflict resolution, abort the merge (`git merge --abort`), check out `main`, record the failure, notify the user, and stop.

#### 3e. Commit the merge resolution

Only needed when conflicts were present:

```bash
git commit --no-edit
```

#### 3f. Push

```bash
git push origin <headRefName>
```

If push is rejected, check out `main`, record the failure, notify the user, and stop.

#### 3g. Post a comment

Template:

```
Processed by the dependabot skill.

Changes applied to this branch:
- `pyproject.toml`: bumped `<package>` from `<old-constraint>` to `>=<target>` (group: `<group>`)
- `uv.lock`: deleted and regenerated via `make venv`
- `.github/workflows/`: updated `<action>` from `<old-version>` to `<new-version>`

Merged latest main. Ready for maintainer review.
```

Rules:
- Omit sections that are empty (no Python entry, no GHA entry).
- If the merge resolved conflicts, add: `Note: conflicts in <files> were resolved automatically.`
- If the branch was already up-to-date with main, omit the "Merged latest main" line.

```bash
gh pr comment <number> --body "<comment body>"
```

#### 3h. Return to main

```bash
git checkout main
```

### Step 4 — Final summary

Print the PR number, package/action name, type (direct / indirect-python / github-actions), and outcome (e.g. "processed and pushed" or "failed: <reason>").

---

## Important constraints

- **Confirm before acting.** Wait for user confirmation after Step 2.
- **Process first, merge second.** Apply the version bump and commit before merging main.
- **Never merge.** Leave approvals and merging to the maintainer.
- **Never force-push.**
- **Preserve constraint operators.** Never change `>=` to `==` or vice versa.
- **Oldest PR first.** Always select the PR with the lowest number.
