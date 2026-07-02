---
name: dependabot
description: Handle open dependabot PRs automatically. Detects all open PRs from the Dependabot bot, classifies each updated dependency as direct (declared in pyproject.toml), indirect-python (transitive), or GitHub Actions, then applies the full unified change set (Python dep bumps + GitHub Actions version bumps) to every PR branch so all branches reach the same final state and merge cleanly in any order. Posts a summary comment on each PR. Merging is left to the maintainer. Use this skill whenever the user says "handle dependabot PRs", "take care of dependabot", "process dependabot updates", "triage dependabot", "deal with dependabot", or anything similar about pending dependabot pull requests.
---

## Overview

Automates the prep work for the weekly dependabot PR triage. Collects all Python dep bumps and GitHub Actions version bumps into a single unified change set and applies it to every PR branch — so all branches reach the same final state and merge cleanly in any order.

**Dependency types:**
- **Direct**: declared in `pyproject.toml` — contributes a Python entry to the change set.
- **Indirect-python**: transitive dep, not in `pyproject.toml` — no own Python entry, but still processed.
- **GitHub Actions**: contributes a GHA entry to the change set.

Every branch (direct, indirect-python, and github-actions) receives the full unified change set: all Python dep bumps applied to `pyproject.toml` + `uv.lock` regenerated + all GHA version bumps applied to `.github/workflows/`.

Merging is left to the maintainer.

---

## Workflow

### Step 1 — Discover open dependabot PRs

```bash
gh pr list --author app/dependabot --state open --json number,title,headRefName,url
```

Stop and report if no PRs are found.

### Step 2 — Classify every PR and collect the unified change set

Extract the package name (token after "Bump", lowercased). Search `pyproject.toml` (`[project].dependencies` and all `[dependency-groups]` keys) for that name (case-insensitive):

- **direct**: found — note the group (`project`, `dev`, `doc`, etc.) and current constraint (e.g. `>=0.0.53`).
- **indirect-python**: Python package, not found.
- **github-actions**: owner/action pattern (e.g. `actions/checkout`).

For each **direct** PR, verify the actual latest version on PyPI:

```bash
curl -s "https://pypi.org/pypi/<package>/json" \
  | python3 -c "import sys, json; d = json.load(sys.stdin); print(d['info']['version'])"
```

Use whichever is higher (PyPI vs dependabot) as the **target version**. Note if PyPI was ahead.

For each **github-actions** PR, confirm the version format as it appears in the workflow files:

```bash
grep -r "<action>" .github/workflows/
```

The YAML often uses a `v` prefix (e.g., `@v4`) even if the PR title says "from 3 to 4".

Build the unified change set:

**Python entries** (from direct PRs):

| Package | Old constraint | Target version | Group | Source PR |
|---------|---------------|----------------|-------|-----------|
| `<pkg-a>` | `>=1.0.0` | `1.1.0` | dev | #N |
| `<pkg-b>` | `>=0.0.53` | `0.0.55` | doc | #M |

**GitHub Actions entries** (from github-actions PRs):

| Action | Old version (as in YAML) | New version | Source PR |
|--------|--------------------------|-------------|-----------|
| `<owner>/<action-a>` | `v3` | `v4` | #N |
| `<owner>/<action-b>` | `v1` | `v2` | #M |

Print the full classification table and unified change set, then ask: *"Ready to process these? Any you'd like to skip?"* Wait for confirmation.

### Step 3 — Process all PR branches

Work through **every** PR (direct, indirect-python, and github-actions) one at a time using the same procedure.

#### 3a. Check out the branch

```bash
git fetch origin <headRefName>
git checkout <headRefName>
```

#### 3b. Apply the Python change set

If the Python change set is non-empty, run `uv add` for every Python entry (regardless of which PR owns it), then regenerate the lock file:

- `[project].dependencies` → `uv add "<package>>=<target-version>"`
- `[dependency-groups] dev` → `uv add --group dev "<package>>=<target-version>"`
- `[dependency-groups] doc` → `uv add --group doc "<package>>=<target-version>"`

Preserve the existing constraint operator. For indirect-python and github-actions branches there is no own entry — apply the other entries anyway.

```bash
rm uv.lock
uv sync --all-groups
```

Delete rather than relying on `uv add`'s incremental update — this produces one clean resolution after all changes are applied together. Skip this entire sub-step if the Python change set is empty.

#### 3c. Apply the GitHub Actions change set

If the GHA change set is non-empty, for each entry replace the old version with the new across all workflow files:

```bash
sed -i '' 's|<action>@<old-version>|<action>@<new-version>|g' .github/workflows/*.yml
```

Run for every entry. The branch's own action (if any) is already bumped by dependabot, so the sed is a no-op for it. Skip this entire sub-step if the GHA change set is empty.

#### 3d. Commit and push

```bash
git add pyproject.toml uv.lock .github/workflows/
git commit -m "chore: apply dependabot updates"
git push origin <headRefName>
```

`git add` only stages files that actually changed, so unmodified files are silently ignored. If `git commit` says "nothing to commit" (edge case), note it in the summary and continue.

#### 3e. Post a comment

Template:

```
Processed by the dependabot skill[, grouped with #N, #M, …].

Changes applied to this branch:
- `pyproject.toml`: bumped `<package>` from `<old-constraint>` to `>=<target>` (group: `<group>`) ← this PR
- `pyproject.toml`: bumped `<package>` from `<old-constraint>` to `>=<target>` (group: `<group>`) ← from #N
- `uv.lock`: deleted and regenerated via `uv sync --all-groups`
- `.github/workflows/`: updated `<action>` from `<old-version>` to `<new-version>` ← from #N

All open dependabot changes were applied together to prevent merge conflicts. Ready for maintainer review.
```

Rules:
- Omit "grouped with" if this is the only PR in the batch.
- Mark this PR's own entry `← this PR`; others `← from #N`. For indirect-python, all Python entries are `← from #N`.
- Omit sections that are empty (no Python entries, or no GHA entries).
- Add a PyPI note if PyPI was ahead: `Note: PyPI latest was <v>, ahead of dependabot's <v> — used <v>.`

```bash
gh pr comment <number> --body "<comment body>"
```

#### 3f. Return to main

```bash
git checkout main
```

### Step 4 — Final summary

Print a table listing each PR number, package/action name, type (direct / indirect-python / github-actions), and action taken (e.g. "full change set applied, pushed" or "nothing to commit").

---

## Important constraints

- **Collect before touching.** Complete the unified change set in Step 2 before checking out any branch.
- **Apply all to all.** Every branch receives the full Python change set and the full GHA change set, regardless of its own type or which entry is "its own".
- **Always commit and push.** Never skip for any branch type.
- **Never merge.** Leave approvals and merging to the maintainer.
- **Never force-push.**
- **One branch at a time.** Checkout → apply → commit → push → return to `main` before the next.
- **Preserve constraint operators.** Never change `>=` to `==` or vice versa.
- **Ask before acting.** Wait for user confirmation after Step 2.
