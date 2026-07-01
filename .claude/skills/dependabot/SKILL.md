---
name: dependabot
description: Handle open dependabot PRs automatically. Detects all open PRs from the Dependabot bot, classifies each updated dependency as direct (declared in pyproject.toml) or indirect (transitive or GitHub Actions), and takes the appropriate action: for indirect deps, enables auto-merge once CI passes; for direct deps, checks out the branch, bumps the version constraint in pyproject.toml via uv, commits, pushes, and enables auto-merge. Use this skill whenever the user says "handle dependabot PRs", "take care of dependabot", "merge dependabot PRs", "process dependabot updates", "triage dependabot", "deal with dependabot", or anything similar about pending dependabot pull requests. Always prefer this skill proactively when dependabot is mentioned in the context of open PRs.
---

## Overview

This skill automates the weekly dependabot PR triage workflow. Dependabot updates `uv.lock` (and GitHub Actions pins) to newer package versions. The critical distinction is:

- **Indirect dependency** (transitive or GitHub Actions): the lock file update alone is sufficient. Just ensure CI passes and merge.
- **Direct dependency** (declared in `pyproject.toml`): the version constraint in `pyproject.toml` must also be bumped so new contributors installing from the manifest get the correct minimum version.

The skill handles both cases end-to-end and uses `--auto` merge so PRs close automatically once CI is green, requiring no further babysitting.

---

## Workflow

### Step 1 — Discover open dependabot PRs

```bash
gh pr list --author app/dependabot --state open \
  --json number,title,headRefName,url,statusCheckRollup
```

If no PRs are returned, report "No open dependabot PRs found." and stop.

Print a brief list of the discovered PRs before doing anything else.

### Step 2 — Classify each PR as direct or indirect

Parse the PR title. Dependabot titles follow one of these formats:

- `Bump <package> from <old-version> to <new-version>` (Python/uv packages)
- `Bump <owner>/<action> from <old-version> to <new-version>` (GitHub Actions)

Extract the **package name** (the token immediately after "Bump", lowercased).

Read `pyproject.toml` and search all dependency arrays (`[project].dependencies` and every key under `[dependency-groups]`) for a line containing that package name. The match is case-insensitive.

- **Found** → direct dependency. Note which group it belongs to (`project`, `dev`, `doc`, etc.) and its current version constraint (operator + version, e.g., `>=0.0.53`).
- **Not found** → indirect or GitHub Actions dependency.

Print a classification table before taking any action, and confirm with the user that they want to proceed:

| PR | Title | Type | Group | Current constraint |
|----|-------|------|-------|--------------------|
| #N | Bump ruff ... | direct | dev | `>=0.15.19` |
| #N | Bump actions/checkout ... | indirect | — | — |

Ask: *"Ready to process these PRs? Any you'd like to skip?"*

Wait for the user's confirmation before proceeding.

### Step 3 — Handle indirect / GitHub Actions PRs

For each **indirect** PR (process these first since they require no branch checkout):

1. Enable auto-merge:
   ```bash
   gh pr merge <number> --squash --auto
   ```
   If the repository does not have auto-merge enabled, wait for CI then merge immediately:
   ```bash
   gh pr checks <number> --watch && gh pr merge <number> --squash
   ```

2. Report: `PR #<number> (<package>): indirect — auto-merge enabled.`

### Step 4 — Handle direct dependency PRs

Work through each **direct** PR one at a time. For each:

#### 4a. Determine the target version

From the PR title `Bump <package> from <old> to <new>`, extract `<new>` as dependabot's suggested version.

Then verify whether a newer release exists on PyPI:

```bash
curl -s "https://pypi.org/pypi/<package>/json" \
  | python3 -c "import sys, json; d = json.load(sys.stdin); print(d['info']['version'])"
```

Compare the PyPI latest with dependabot's `<new>`. Use whichever is higher as the **target version**. If PyPI is ahead, note it and use the PyPI version — dependabot may be slightly behind.

(Skip this check for GitHub Actions PRs; they are not PyPI packages.)

#### 4b. Check out the branch

```bash
git fetch origin <headRefName>
git checkout <headRefName>
```

#### 4c. Build the uv add command

Use the group you identified in Step 2 to pick the right flag:

- `[project].dependencies` → no group flag: `uv add "<package>>=<new-version>"`
- `[dependency-groups] dev` → `uv add --group dev "<package>>=<new-version>"`
- `[dependency-groups] doc` → `uv add --group doc "<package>>=<new-version>"`
- Any other group → `uv add --group <group> "<package>>=<new-version>"`

Always use `>=` as the operator unless the existing constraint uses a different operator (`==`, `~=`, etc.), in which case preserve it. The goal is to update the lower bound to the new version while keeping the constraint style unchanged.

#### 4d. Run the update

```bash
uv add [--group <group>] "<package>>=<new-version>"
```

This updates `pyproject.toml` in place and may touch `uv.lock`. Verify the change:

```bash
grep -n "<package>" pyproject.toml
```

Confirm that the version number in `pyproject.toml` now reflects `<new-version>`.

#### 4e. Regenerate the lock file and commit

After updating `pyproject.toml`, delete the existing lock file and do a clean regeneration. This ensures the lock is produced by our own resolver rather than being a partial merge of dependabot's lock and our toolchain:

```bash
rm uv.lock
uv sync --all-groups
```

Then commit and push:

```bash
git add pyproject.toml uv.lock
git commit -m "chore: bump <package> to <target-version> in pyproject.toml"
git push origin <headRefName>
```

#### 4f. Enable auto-merge

```bash
gh pr merge <number> --squash --auto
```

If auto-merge is unavailable:
```bash
gh pr checks <number> --watch && gh pr merge <number> --squash
```

#### 4g. Return to main before handling the next PR

```bash
git checkout main
```

Report: `PR #<number> (<package>): direct — pyproject.toml updated to >=<new-version>, pushed, auto-merge enabled.`

### Step 5 — Final summary

After all PRs are processed, print a clean summary:

| PR | Package | Type | Action |
|----|---------|------|--------|
| #N | ruff | direct | pyproject.toml bumped to >=0.15.20, auto-merge enabled |
| #N | actions/checkout | indirect | auto-merge enabled |
| … | … | … | … |

---

## Important constraints

- **Never merge without CI.** Always use `--auto` so the merge only happens once tests pass. Never use `gh pr merge` without `--auto` unless you explicitly waited via `gh pr checks --watch`.
- **Never force-push.** Dependabot branches are shared; only regular pushes are allowed.
- **One PR at a time for direct deps.** Check out, update, commit, push, and return to `main` before moving to the next direct-dep PR. Never interleave checkouts.
- **Preserve the constraint operator.** If `pyproject.toml` has `>=old`, write `>=new`. If it has `==old`, write `==new`. Never silently change the operator style.
- **Always return to `main` when done.** Run `git checkout main` after the last PR is processed.
- **Ask before acting.** Show the classification table from Step 2 and wait for user confirmation before making any changes.
