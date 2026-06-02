---
name: create-pr
description: Opens a pull request for the current branch on the GLIDE repository. Use this skill whenever the user says "create a PR", "open a PR", "submit this for review", "make a pull request", or asks to push and PR a branch — even if they haven't finished working on it yet. The skill pushes the branch if needed, fills in the PR template from the diff and user inputs, updates CHANGELOG if required, attaches the right labels, ticks the appropriate checklist boxes, and creates the PR via gh. Invoke even when the request is casual ("can you open the PR for me?").
---

## Overview

This skill creates a pull request on the GLIDE repository for the branch the user is currently on. It produces a correctly filled-in PR template, updates the changelog if the change is user-facing, attaches accurate labels, and calls `gh pr create` to open the PR.

The user may create the PR before they are ready for review (e.g., to share a work-in-progress). The skill handles both cases: a fully ready PR with all checklist boxes ticked, and a draft/partial PR with only the boxes that genuinely apply.

---

## Workflow

### Step 1 — Collect required context

Run these in parallel to build a picture of the branch:

```bash
git log main..HEAD --oneline
git diff main...HEAD --stat
git diff main...HEAD
gh issue list --limit 30 --state open
```

Also read `CHANGELOG.md` (the first 30 lines suffice) and `.github/PULL_REQUEST_TEMPLATE.md`.

Before proceeding, you need:
1. **The issue or ticket this PR addresses.** If the user did not mention one, ask: *"Which issue or ticket does this branch address? If it's a GitHub issue, give me the number; if it's a project board item, paste the link. If there's no issue, just say so."* Do not proceed until you have this answer.
2. **Noteworthy implementation decisions or trade-offs.** Ask: *"Is there anything noteworthy about implementation decisions or trade-offs you want captured in the PR description?"* An empty answer is fine — do not fill this section yourself.

Ask both questions in one message and wait for the user's reply.

### Step 2 — Determine the PR title

Derive the title from the issue/ticket the user pointed to. Fetch the issue body if it's a GitHub issue (`gh issue view <N>`). Aim for a short imperative title (under 70 characters) that describes what the PR *does*, consistent with the pattern in recent PRs (e.g., "Add constrained optimization to ActiveSampler", "Fix decision tree figure").

If the branch only partially addresses the ticket, prefix the title with "WIP:" to signal it is not yet complete.

### Step 3 — Decide whether a CHANGELOG update is needed

A CHANGELOG update is needed when the change is **user-facing**: new public API, changed behaviour, removed or renamed public symbols, or fixed bugs that affected users. It is not needed for pure refactoring, CI changes, internal reorganisation, or documentation-only changes.

If a CHANGELOG update is needed and the user has not already added one:
1. Add a bullet to the correct section (`Added`, `Changed`, or `Fixed`) under `## [Next release]` in `CHANGELOG.md`. Keep it concise and user-facing (what changed, not how).
2. You will check the CHANGELOG checklist box later.

If the user already updated CHANGELOG, just verify the entry is under `## [Next release]`.

### Step 4 — Fill in the PR template

Produce the PR body using this exact structure:

```
### Description

- What does this PR do?
<one or two sentences from the diff — clear, concrete, no waffle>
- Which issue does it close? (use `Closes #<number>`)
<see guidance below>
- Any noteworthy implementation decisions or trade-offs?
<only if the user provided something — otherwise leave the line blank>

### Checklist

Quality gates that must be satisfied before requesting a review:

- [x] I have read `CONTRIBUTING.md`
- [x or space] `make lint` passes
- [x or space] `make type-check` passes
- [x or space] `make tests` passes
- [x or space] `make coverage` reports 100% coverage
- [x or space] `make test-notebooks` passes
- [x or space] New public API has numpy-style docstrings
- [x or space] New public API is inserted in the API reference section of the documentation
- [x or space] Docs build without warnings (`make doc`)
- [x or space] `CHANGELOG.md` updated if the change is user-facing

### LLM usage

Disclose if an LLM was used in writing this PR:
- [ ] No LLM used
- [x] I used an LLM and I went through and validated all the code myself
```

**Issue linking:**
- GitHub issue → `Closes #<N>`
- Project board item only → `Handles this [ticket](<url>)` (same style as PR #233)
- Both → `Handles this [ticket](<url>) Closes #<N>`
- No issue → leave the bullet empty or write "N/A"

**Checklist logic — only tick a box when you are confident it applies:**

| Box | Tick when |
|---|---|
| `make lint` | The diff contains only formatting/trivial fixes, or the user confirms it passes |
| `make type-check` | No new type annotations or complex signatures were added, or the user confirms it passes |
| `make tests` | The diff contains test changes that clearly cover the new code, or the user confirms it passes |
| `make coverage` | Same as tests |
| `make test-notebooks` | Notebooks were not modified, or the user confirms it passes |
| New public API docstrings | New public methods/classes were added and docstrings are visible in the diff |
| API reference | New public API was added and the relevant `docs/api_reference` page was updated in the diff |
| Docs build | Documentation files were modified, or docs are untouched |
| CHANGELOG updated | A user-facing change was made and you (or the user) added a CHANGELOG entry |

When in doubt, leave the box unchecked. It is better to leave boxes unticked than to tick boxes that the user hasn't verified. The user can tick the remaining boxes before requesting review.

Always check "I have read `CONTRIBUTING.md`" and always check the LLM box (since this skill uses Claude).

### Step 5 — Determine labels

Apply one **type label** and one or more **component labels** as applicable.

**Type labels (pick one):**

| Label | When |
|---|---|
| `feature` | New public API, new algorithm, new capability |
| `bug` | Fixes incorrect behaviour |
| `refactoring` | Code restructuring with no functional change |
| `documentation` | Only docs, docstrings, notebooks, or user guide changes |
| `repository` | CI, tooling, Makefile, GitHub config, dependencies |

**Component labels (pick all that apply):**

| Label | When |
|---|---|
| `PPI` | Touches prediction-powered inference estimators or related code |
| `ASI` | Touches active statistical inference |
| `classical` | Touches classical inference estimators |
| `PTD` | Touches Predict-Then-Debias estimators |
| `Stratified PPI` | Touches StratifiedPPIMeanEstimator |
| `Stratified PTD` | Touches StratifiedPTDMeanEstimator |
| `Stratified sampler` | Touches StratifiedSampler |
| `Active sampler` | Touches ActiveSampler |
| `Cost optimal sampler` | Touches CostOptimalSampler |
| `Cost optimal random sampler` | Touches CostOptimalRandomSampler |
| `Dataset` | Touches simulators or dataset generators |
| `inference result` | Touches MeanInferenceResult or ConfidenceInterval classes |
| `Breaking change` | Public API removed or signature changed in a breaking way |

Derive component labels from the diff: look at which files under `glide/` were modified.

### Step 6 — Determine reviewer

The typical reviewers are `gmartinon-ed`, `imerad`, and `mraki-ed`. Pick the one(s) that are not the PR author. If you know who the author is from `git config user.name` or recent commits, exclude them and assign the others. If uncertain, ask the user.

### Step 7 — Push and create the PR

First, check if the branch is already on the remote:

```bash
git ls-remote --heads origin HEAD_BRANCH_NAME
```

If not pushed yet:
```bash
git push -u origin HEAD
```

Then create the PR:

```bash
gh pr create \
  --title "<title>" \
  --body "$(cat <<'EOF'
<filled body>
EOF
)" \
  --label "<label1>" --label "<label2>" \
  --reviewer "<reviewer>"
```

After creation, print the PR URL for the user.

If you updated CHANGELOG, also tell the user: *"I updated CHANGELOG.md — you'll want to commit that change before merging."*

---

## Important constraints

- Never fill in "Any noteworthy implementation decisions or trade-offs?" yourself based on the diff. Only include content the user explicitly provided.
- Never tick checklist boxes you cannot verify from the diff alone (e.g., lint, tests, coverage). Leave them for the user.
- If the branch only partially addresses the ticket, say so clearly in the PR description and prefix the title with "WIP:".
- Always check the LLM box, since Claude is writing this PR.
- Use `Closes #N` only for GitHub issues that this PR fully resolves. Use `Handles this [ticket](url)` for partial work or project-board-only items.
