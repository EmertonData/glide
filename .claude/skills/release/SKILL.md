---
name: release
description: Automates the GLIDE release process end-to-end: bumps the version, edits CHANGELOG, opens a release PR, triggers TestPyPI, creates the git tag, and creates the GitHub release. Use whenever the user says "make a release", "release a new version", "ship a new version", "publish a release", "publish to PyPI", or invokes /release with an optional bump type (major/minor/patch).
---

## Overview

This skill drives the full GLIDE release process with minimal human action. There are two human gates: merging the release PR (after CI passes) and confirming the TestPyPI installation. Everything else is automated.

The skill runs as a single linear flow within one conversation. When it reaches the merge gate, it pauses and explicitly tells the user to come back to this same conversation after merging.

---

## Step 1 — Check for a clean working tree

```bash
git status --short
```

If there are any uncommitted or untracked changes, warn the user and stop until they resolve it.

## Step 2 — Determine bump type

If the user passed `major`, `minor`, or `patch` as an argument to the skill, use that. Otherwise ask:

> "Which version bump does this release require: `major`, `minor`, or `patch`?"

## Step 3 — Pull latest main

```bash
git checkout main && git pull origin main
```

## Step 4 — Bump version files

```bash
make bump-minor   # or bump-major or bump-patch
```

Read `NEW_VERSION` from the command output. Use it in all subsequent steps.

## Step 5 — Create the release branch

```bash
git checkout -b release/v<NEW_VERSION>
```

## Step 6 — Update CHANGELOG.md

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

4. **Simplify the new changelog section** before showing it to the user:
   - Combine entries that refer to the same feature into a single bullet point.
   - Remove any remaining duplicates.
   - Rewrite technical or jargon-heavy entries into plain, user-friendly language — focus on what changed and why it matters to the user, not implementation details.
   - Drop empty subsections (e.g., a `### 🐛 Fixed` heading with no bullets).

   Present the simplified section to the user and ask:

   > "Here is the changelog section for `vX.Y.Z`:
   >
   > <SECTION CONTENT>
   >
   > Does this look good, or would you like any changes?"

   Wait for approval. Apply any requested edits and re-present until the user confirms. Then write the final version to disk.

## Step 7 — Commit, push, and open a PR

```bash
git add pyproject.toml CITATION.cff CHANGELOG.md
git commit -m "chore: bump version to v<NEW_VERSION> and update changelog"
git push -u origin release/v<NEW_VERSION>
```

Ask the user:

> "Do you have a release ticket URL to link in the PR? (paste it or say no)"

Then invoke the `create-pr` skill with the following context:

- **Title:** `Release v<NEW_VERSION>`
- **Body:** the approved changelog section for `[X.Y.Z]` verbatim (the lines moved in Step 6). If the user provided a ticket URL, include a `Closes <URL>` line; otherwise omit it.
- **Labels:** `release`
- Do **not** assign reviewers.

Once the PR is open, say:

> "The release PR is open at <URL>. Once CI passes, merge it, then **come back to this conversation** and tell me it's merged so I can continue with tagging and publishing."

**Stop here and wait.** Do not proceed until the user returns to this conversation and confirms the PR has been merged.

---

## Step 8 — Confirm merge and proceed

When the user returns and says the PR is merged, confirm by checking:

```bash
REPO_URL=$(gh repo view --json url -q .url)
git fetch origin
```

Then say:

> "I'll now:
> 1. Trigger the TestPyPI workflow
> 2. Create and push the tag after you confirm the TestPyPI build
> 3. Create the GitHub release
>
> Shall I proceed?"

Wait for confirmation before continuing.

## Step 9 — Trigger and monitor the TestPyPI workflow

Trigger the workflow, wait a few seconds for it to register, then grab the run ID and watch it to completion:

```bash
gh workflow run release.yml --ref main
sleep 5
RUN_ID=$(gh run list --workflow=release.yml --limit 1 --json databaseId -q '.[0].databaseId')
gh run watch "$RUN_ID" --exit-status
```

`gh run watch` streams live output and exits non-zero if the run fails. If it fails, stop and show the user the failure URL (`${REPO_URL}/actions/runs/${RUN_ID}`) before proceeding.

## Step 10 — Verify TestPyPI installation

Once the workflow passes, verify the package is installable from TestPyPI:

```bash
uv venv /tmp/glide-testpypi-test --python 3.12
/tmp/glide-testpypi-test/bin/uv pip install \
  --index-url https://test.pypi.org/simple/ --no-deps glide-py
INSTALLED=$(/tmp/glide-testpypi-test/bin/python -c "import glide; print(glide.__version__)")
echo "Installed version: $INSTALLED"
```

Check that `$INSTALLED` matches `v<NEW_VERSION>`. If it does not match or the install fails, stop and report before proceeding.

## Step 11 — Pull latest main and create the tag

```bash
git checkout main && git pull origin main
git tag v<NEW_VERSION>
git push origin v<NEW_VERSION>
```

Tell the user the tag has been pushed and the PyPI publish workflow is now running.

## Step 12 — Monitor the PyPI publish workflow

Poll until the tag-triggered workflow run appears (timeout after 60 seconds), then watch it:

```bash
DEADLINE=$((SECONDS + 60))
until RUN_ID=$(gh run list --workflow=release.yml --branch v<NEW_VERSION> --limit 1 --json databaseId -q '.[0].databaseId') && [ -n "$RUN_ID" ]; do
  [ $SECONDS -ge $DEADLINE ] && { echo "Timed out waiting for workflow run."; exit 1; }
  sleep 5
done
gh run watch "$RUN_ID" --exit-status
```

If it fails, report the failure URL (`${REPO_URL}/actions/runs/${RUN_ID}`) and stop.

## Step 13 — Verify PyPI installation

Once the publish workflow passes, verify the package is installable from PyPI:

```bash
uv venv /tmp/glide-pypi-test --python 3.12
uv cache clean
/tmp/glide-pypi-test/bin/uv pip install glide-py
INSTALLED=$(/tmp/glide-pypi-test/bin/python -c "import glide; print(glide.__version__)")
echo "Installed version: $INSTALLED"
```

Check that `$INSTALLED` matches `v<NEW_VERSION>`. If it does not, report and stop.

## Step 14 — Create the GitHub release

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
- Derive the GitHub repo URL dynamically with `gh repo view --json url -q .url` — never hardcode it.
