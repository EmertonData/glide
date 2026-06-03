---
name: create-pr
description: Opens a pull request for the current branch on the GLIDE repository. Use this skill whenever the user says "create a PR", "open a PR", "submit this for review", "make a pull request", or asks to push and PR a branch — even if they haven't finished working on it yet. The skill pushes the branch if needed, fills in the PR template from the diff and user inputs, updates CHANGELOG if required, attaches the right labels, ticks the appropriate checklist boxes, and creates the PR via gh. Invoke even when the request is casual ("can you open the PR for me?").
---

## Overview

This skill creates a pull request on the GLIDE repository for the branch the user is currently on. It produces a correctly filled-in PR template, updates the changelog if the change is user-facing, attaches accurate labels, and calls `gh pr create` to open the PR.

> **Note:** Some instructions here reference specific template sections and checklist items as they existed when this skill was written. If you notice a mismatch between these instructions and the actual `.github/PULL_REQUEST_TEMPLATE.md`, let the user know before creating the PR.

The user may create the PR before they are ready for review (e.g., to share a work-in-progress). The skill handles both cases: a fully ready PR with all checklist boxes ticked, and a draft/partial PR with only the boxes that genuinely apply.

---

## Workflow

### Step 1 — Collect required context

Always run these first:

```bash
git log main..HEAD --oneline
git diff main...HEAD --stat
gh issue list --limit 30 --state open
```

Also read `CHANGELOG.md` (the first 30 lines suffice) and `.github/PULL_REQUEST_TEMPLATE.md`.

**Reading the full diff** — only do this if you genuinely need more detail to write the PR description or determine labels. Skip it if:
- The user already described what the branch does in enough detail, or
- The `--stat` output makes the scope of changes obvious.

If you do need the full diff, first check how large it is from the `--stat` summary (total insertions + deletions). If it exceeds ~300 lines, ask the user before loading it: *"The diff is large (~N lines). Do you want me to read through it, or can you tell me what the branch does?"*

Before proceeding, you need two things from the user. Ask them together in one message and wait for the reply:
1. **The issue or ticket this PR addresses.** *"Which issue or ticket does this branch address? If it's a GitHub issue, give me the number; if it's a project board item, paste the link. If there's no issue, just say so."*
2. **Noteworthy implementation decisions or trade-offs.** *"Is there anything noteworthy about implementation decisions or trade-offs you want captured in the PR description?"* An empty answer is fine — do not fill this section yourself.

### Step 2 — Determine the PR title

Derive the title from the issue/ticket the user pointed to. Fetch the issue body if it's a GitHub issue (`gh issue view <N>`). Aim for a short imperative title (under 70 characters) that describes what the PR *does*, consistent with the pattern in recent PRs (e.g., "Add constrained optimization to ActiveSampler", "Fix decision tree figure").

### Step 3 — Decide whether a CHANGELOG update is needed

A CHANGELOG update is needed when the change is **user-facing**: new public API, changed behaviour, removed or renamed public symbols, or fixed bugs that affected users. It is not needed for pure refactoring, CI changes or internal reorganisation changes.

If a CHANGELOG update is needed and the user has not already added one:
1. Add a bullet to the correct section (`Added`, `Changed`, or `Fixed`) under `## [Next release]` in `CHANGELOG.md`. Keep it concise and user-facing (what changed, not how).
2. You will check the CHANGELOG checklist box later.

If the user already updated CHANGELOG, just verify the entry is under `## [Next release]`.

If necessary, update the contributors section under `## [Next release]` with user's github username.

### Step 4 — Fill in the PR template

Read `.github/PULL_REQUEST_TEMPLATE.md` to get the current template structure, then fill it in as follows.

**Description section:** Write one or two concrete sentences summarising what the PR does, derived from the diff. Link the issue using the guidance below. Include noteworthy implementation decisions only if the user explicitly provided them — do not infer or fill this yourself.

**Issue linking:**
- GitHub issue → `Closes #<N>`
- Project board item only → `Handles this [ticket](<url>)` (same style as PR #233)
- Both → `Handles this [ticket](<url>) Closes #<N>`
- No issue → leave blank or write "N/A"

**Checklist:** Do not run quality-gate commands yourself — the user is responsible for those. Use the following rules to decide which boxes to tick:

- **Lint, type-check, coverage, docstrings, docs build:** tick if the diff touches Python code; leave unchecked for docs-only or config-only changes.
- **API Reference enriched:** tick if the PR adds or modifies public API (new classes, functions, or changed signatures). If this box applies but cannot be ticked because the API reference has not been updated, tell the user explicitly and ask them to add the missing documentation before the PR is merged.
- **New estimator PRs:** a PR adding a new estimator is only complete when it also includes a scientific validation notebook, a user guide section, a tutorial, and a README update linking the relevant literature. If any of these are missing, flag it to the user before creating the PR.

Always check "I have read `CONTRIBUTING.md`" and always check the LLM box (since this skill uses Claude).

### Step 5 — Determine labels

Fetch the current label list from GitHub:

```bash
gh label list --limit 100
```

From the output, pick all labels that apply to this PR. Use the label names and descriptions returned by the CLI to decide relevance — do not rely on hardcoded assumptions. Apply every label that fits; there is no cap.

### Step 6 — Push and create the PR

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
  --label "<label1>" --label "<label2>"
```

Do not assign reviewers — the user will request reviews themselves when ready.

After creation, print the PR URL for the user.

If you updated CHANGELOG, also tell the user: *"I updated CHANGELOG.md — you'll want to commit that change before merging."*

---

## Important constraints

- Never fill in "Any noteworthy implementation decisions or trade-offs?" yourself based on the diff. Only include content the user explicitly provided.
- Always check the "read `CONTRIBUTING.md`" box.
- Always check the LLM box, since Claude is writing this PR.
- Use `Closes #N` for GitHub issues. Use `Handles this [ticket](url)` for project-board-only items.
