---
name: dependabot
description: Handle open dependabot PRs automatically. Detects all open PRs from the Dependabot bot, classifies each updated dependency as direct (declared in pyproject.toml) or indirect (transitive or GitHub Actions), and takes the appropriate action: for indirect deps, reports them as ready for maintainer review; for direct deps, checks out the branch, bumps the version constraint in pyproject.toml via uv, regenerates uv.lock, and pushes — so the maintainer only needs to approve and click merge. Use this skill whenever the user says "handle dependabot PRs", "take care of dependabot", "process dependabot updates", "triage dependabot", "deal with dependabot", or anything similar about pending dependabot pull requests.
---

## Overview

This skill automates the prep work for the weekly dependabot PR triage. The key distinction:

- **Indirect dependency** (transitive or GitHub Actions): the lock file update is sufficient. Nothing to do — just flag it as ready for maintainer review.
- **Direct dependency** (declared in `pyproject.toml`): the version constraint there must also be bumped so new contributors get the correct minimum version. The skill checks out the branch, updates `pyproject.toml`, regenerates `uv.lock`, and pushes.

Merging is left to the maintainer in both cases.

---

## Workflow

### Step 1 — Discover open dependabot PRs

```bash
gh pr list --author app/dependabot --state open \
  --json number,title,headRefName,url,statusCheckRollup
```

If no PRs are returned, report "No open dependabot PRs found." and stop.

### Step 2 — Classify each PR as direct or indirect

Parse the PR title. Dependabot titles follow one of these formats:

- `Bump <package> from <old-version> to <new-version>` (Python/uv packages)
- `Bump <owner>/<action> from <old-version> to <new-version>` (GitHub Actions)

Extract the **package name** (the token immediately after "Bump", lowercased).

Read `pyproject.toml` and search all dependency arrays (`[project].dependencies` and every key under `[dependency-groups]`) for a line containing that package name (case-insensitive).

- **Found** → direct dependency. Note which group (`project`, `dev`, `doc`, etc.) and its current version constraint (e.g., `>=0.0.53`).
- **Not found** → indirect or GitHub Actions dependency.

Print a classification table and ask the user to confirm before proceeding:

| PR | Title | Type | Group | Current constraint |
|----|-------|------|-------|--------------------|
| #N | Bump ruff ... | direct | dev | `>=0.15.19` |
| #N | Bump actions/checkout ... | indirect | — | — |

Ask: *"Ready to process these? Any you'd like to skip?"* Wait for confirmation.

### Step 3 — Handle indirect / GitHub Actions PRs

Nothing to change on these branches. Report each one:

`PR #<number> (<package>): indirect — no changes needed, ready for maintainer review.`

### Step 4 — Handle direct dependency PRs

Work through each **direct** PR one at a time.

#### 4a. Determine the target version

From the PR title, extract dependabot's suggested version. Then verify the actual latest on PyPI:

```bash
curl -s "https://pypi.org/pypi/<package>/json" \
  | python3 -c "import sys, json; d = json.load(sys.stdin); print(d['info']['version'])"
```

Use whichever is higher as the **target version**. If PyPI is ahead of dependabot, note it.

#### 4b. Check out the branch

```bash
git fetch origin <headRefName>
git checkout <headRefName>
```

#### 4c. Update pyproject.toml

Use the group from Step 2:

- `[project].dependencies` → `uv add "<package>>=<target-version>"`
- `[dependency-groups] dev` → `uv add --group dev "<package>>=<target-version>"`
- `[dependency-groups] doc` → `uv add --group doc "<package>>=<target-version>"`

Preserve the existing constraint operator. If the constraint was `>=old`, use `>=new`. If `==old`, use `==new`.

Verify the change was applied:

```bash
grep -n "<package>" pyproject.toml
```

#### 4d. Regenerate the lock file

```bash
rm uv.lock
uv sync --all-groups
```

#### 4e. Commit and push

```bash
git add pyproject.toml uv.lock
git commit -m "chore: bump <package> to <target-version> in pyproject.toml"
git push origin <headRefName>
```

#### 4f. Return to main

```bash
git checkout main
```

Report: `PR #<number> (<package>): direct — pyproject.toml bumped to >=<target-version>, pushed. Ready for maintainer review.`

### Step 5 — Final summary

| PR | Package | Type | Action |
|----|---------|------|--------|
| #N | ruff | direct | pyproject.toml bumped to >=0.15.20, pushed |
| #N | actions/checkout | indirect | no changes needed |
| … | … | … | … |

Tell the user which PRs are ready for their review and approval.

---

## Important constraints

- **Never merge.** Leave approvals and merging entirely to the maintainer.
- **Never force-push.** Only regular pushes on dependabot branches.
- **One PR at a time for direct deps.** Checkout, update, commit, push, return to `main` before the next one.
- **Preserve the constraint operator.** Never silently change `>=` to `==` or vice versa.
- **Always return to `main` when done.**
- **Ask before acting.** Show the classification table and wait for user confirmation.
