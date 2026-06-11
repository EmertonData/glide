---
name: release
description: Automates the GLIDE release process end-to-end: bumps the version, edits CHANGELOG, opens a release PR, triggers TestPyPI, creates the git tag, and creates the GitHub release. Use whenever the user says "make a release", "cut a release", "release a new version", "ship a new version", "publish a release", "tag the release", "publish to PyPI", or invokes /release with an optional bump type (major/minor/patch). Invoke even when phrased informally ("time to ship", "let's cut a patch", "push a new version").
---

## Overview

This skill drives the full GLIDE release process with minimal human action. There are two human gates: merging the release PR (after CI passes) and confirming the TestPyPI installation. Everything else is automated.

The skill detects where in the process things stand and picks up from the right phase.

---

## Step 1 — Detect current state

Run these commands to understand current state:

```bash
git fetch origin
git describe --tags --abbrev=0 2>/dev/null || echo "NO_TAG"
gh pr list --state open --head "release/" --json title,url,headRefName
git show origin/main:pyproject.toml | head -5
```

**Routing logic:**

- If there is an open PR whose head branch starts with `release/`: the release branch was created but the PR is not yet merged → tell the user to merge it and re-invoke. Do not proceed further.
- If `origin/main` has a version **newer** than the latest tag, OR the latest tag is `NO_TAG` and `origin/main` has any non-zero version: the release PR has already been merged → go to **Phase B**.
- Otherwise: no release is in flight → go to **Phase A**.

---

## Phase A — Create the release branch

### A0. Confirm all intended changes are on main

Before touching anything, ask the user:

> "Before I start: please confirm that `main` contains all the changes intended for this release. Any PR that isn't merged yet won't be included. Good to go?"

Wait for confirmation.

### A1. Check for a clean working tree

```bash
git status --short
```

If there are any uncommitted or untracked changes, warn the user and stop until they resolve it.

### A2. Determine bump type

If the user passed `major`, `minor`, or `patch` as an argument to the skill, use that. Otherwise ask:

> "Which version bump does this release require: `major`, `minor`, or `patch`?"

### A3. Pull latest main

```bash
git checkout main && git pull origin main
```

### A4. Compute the new version without modifying files

Read the current version from `pyproject.toml` (the `version = "X.Y.Z"` line under `[project]`) and derive the new version by incrementing the appropriate component. Do this in your head — do not run any bump command yet.

### A5. Create the release branch

```bash
git checkout -b release/v<NEW_VERSION>
```

### A6. Bump version files

```bash
make bump-major   # or bump-minor or bump-patch
```

Verify the new version in `pyproject.toml` matches what you computed in A4. If it doesn't match, stop and report.

### A7. Update CHANGELOG.md

Read the full current `CHANGELOG.md`. Then:

1. **Move the `[Next release]` content** into a new versioned section immediately below the `[Next release]` header block. The new section header must be:
   ```
   ## [X.Y.Z] – YYYY-MM-DD
   ```
   Use today's actual date (`date +%Y-%m-%d`).

2. **Gather contributors** from git log since the last tag:
   ```bash
   git log --no-merges --format="%an" $(git describe --tags --abbrev=0 2>/dev/null)..HEAD | sort -u
   ```
   Format each name as `@<name>` (replacing spaces with hyphens if needed). Fill in the `💛 Contributors` line under the new section. If the previous release's contributors section in `CHANGELOG.md` shows a different username format, match that format.

3. **Reset the `[Next release]` section** at the top, leaving it empty:
   ```markdown
   ## [Next release]

   ### ✨ Added

   ### 🔄 Changed

   ### 🐛 Fixed

   ### 💛 Contributors
   ```

Write the updated file back to disk.

### A8. Commit

```bash
git add pyproject.toml CITATION.cff CHANGELOG.md
git commit -m "chore: bump version to v<NEW_VERSION> and update changelog"
```

### A9. Push and open a PR

Push the branch:

```bash
git push -u origin release/v<NEW_VERSION>
```

Fetch the repo URL dynamically:

```bash
gh repo view --json url -q .url
```

Create the PR with `gh pr create`:

- **Title:** `Release v<NEW_VERSION>`
- **Body:** paste the changelog section for `[X.Y.Z]` verbatim (the lines you just moved). No issue to close — omit the `Closes` line.
- **Labels:** attach the `release` label if it exists (`gh label list --limit 100`); otherwise no label.
- Do **not** assign reviewers.

Print the PR URL for the user, then say:

> "The release PR is open at <URL>. Once CI passes, merge it, then run `/release` again to continue with tagging and publishing."

---

## Phase B — Tag, publish, and create GitHub release

This phase runs after the release PR has been merged into `main`.

### B1. Confirm with user

Fetch the repo URL:

```bash
REPO_URL=$(gh repo view --json url -q .url)
```

Show the user:

> "Detected version `vX.Y.Z` on `main` (previous tag: `vA.B.C`). I'll now:
> 1. Trigger the TestPyPI workflow
> 2. Create and push the tag after you confirm the TestPyPI build
> 3. Create the GitHub release
>
> Shall I proceed?"

Wait for confirmation before continuing.

### B2. Trigger and monitor the TestPyPI workflow

Trigger the workflow, wait a few seconds for it to register, then grab the run ID and watch it to completion:

```bash
gh workflow run release.yml --ref main
sleep 5
RUN_ID=$(gh run list --workflow=release.yml --limit 1 --json databaseId -q '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```

`gh run watch` streams live output and exits non-zero if the run fails. If it fails, stop and show the user the failure URL (`${REPO_URL}/actions/runs/${RUN_ID}`) before proceeding.

### B3. Verify TestPyPI installation

Once the workflow passes, verify the package is installable from TestPyPI:

```bash
uv venv /tmp/glide-testpypi-test --python 3.12
/tmp/glide-testpypi-test/bin/uv pip install \
  --index-url https://test.pypi.org/simple/ --no-deps glide-py
INSTALLED=$(/tmp/glide-testpypi-test/bin/python -c "import glide; print(glide.__version__)")
echo "Installed version: $INSTALLED"
```

Check that `$INSTALLED` matches `v<NEW_VERSION>`. If it does not match or the install fails, stop and report before proceeding.

### B4. Pull latest main and create the tag

```bash
git checkout main && git pull origin main
git tag v<NEW_VERSION>
git push origin v<NEW_VERSION>
```

Tell the user the tag has been pushed and the PyPI publish workflow is now running.

### B5. Monitor the PyPI publish workflow

Wait for the tag-triggered workflow run to appear, then watch it:

```bash
sleep 10
RUN_ID=$(gh run list --workflow=release.yml --limit 1 --json databaseId -q '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```

If it fails, report the failure URL (`${REPO_URL}/actions/runs/${RUN_ID}`) and stop.

### B6. Verify PyPI installation

Once the publish workflow passes, verify the package is installable from PyPI:

```bash
uv venv /tmp/glide-pypi-test --python 3.12
uv cache clean
/tmp/glide-pypi-test/bin/uv pip install glide-py
INSTALLED=$(/tmp/glide-pypi-test/bin/python -c "import glide; print(glide.__version__)")
echo "Installed version: $INSTALLED"
```

Check that `$INSTALLED` matches `v<NEW_VERSION>`. If it does not, report and stop.

### B7. Create the GitHub release

Extract the relevant section from `CHANGELOG.md` on `main` (the `## [X.Y.Z]` block — everything between that header and the next `## [` header).

```bash
gh release create v<NEW_VERSION> \
  --title "v<NEW_VERSION>" \
  --notes "$(cat <<'EOF'
<CHANGELOG SECTION CONTENT>
EOF
)" \
  --latest
```

Print the release URL and report that the release is complete.

---

## Important constraints

- Never push to `main` directly. All changes to `main` go through the release PR.
- Never create a tag before both the TestPyPI workflow has passed (verified via `gh run watch`) and the installation from TestPyPI has succeeded.
- If any command fails (e.g., `make bump-*`, `gh workflow run`), stop and report the error to the user before proceeding.
- The workflow file is `release.yml` — use this exact filename with `gh workflow run`.
- When re-invoked after a partial Phase A, always re-run Step 1 state detection to avoid creating duplicate branches or tags.
- Derive the GitHub repo URL dynamically with `gh repo view --json url -q .url` — never hardcode it.
