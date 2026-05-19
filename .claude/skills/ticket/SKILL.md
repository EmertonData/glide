---
name: ticket
description: Writes a developer-ready GitHub ticket for the GLIDE dev team, from any input: research paper (arXiv, PDF, screenshots, reference implementation), refactoring request, documentation task, or repository maintenance work. Always use this skill when the user says "write a ticket", "create a ticket", "make a ticket for", or describes something they want the dev team to implement, refactor, document, or clean up. The skill explores the GLIDE codebase, proposes architecture choices, writes concrete pseudocode, and produces a markdown file in tickets/. Invoke even when the request is informal or partial.
---

## Overview

This skill produces a single markdown file in `tickets/` that a junior Python developer can pick up and act on — without having read any paper and without a statistics background. Every concept gets explained, every term gets defined.

There are four ticket types, each with its own template. The workflow is the same for all of them.

---

## Workflow

### 1. Classify the ticket

| Type | When to use |
|---|---|
| **Feature** | New algorithm, estimator, or capability. Often but not always from a paper. |
| **Refactoring** | Restructuring, renaming, or reorganising existing code without changing behaviour. |
| **Documentation** | New or improved docstrings, user guides, notebooks, or API reference. |
| **Repository** | CI/CD, tooling, dependency updates, test infrastructure, or any non-functional improvement. |

### 2. Gather inputs

Collect everything the user provided before writing anything:

- **Paper link** (arXiv, DOI, URL) — fetch it; read abstract, main theorem, algorithm box, and any worked examples.
- **Screenshots** (formulas, pseudocode, paragraphs) — read them.
- **Reference implementation** (R or Python from the authors) — read it.
- **Code pointers** — read the files mentioned.
- **Scope notes** — which parts to cover, which to skip, specific cases or generalisations.

For **Feature** tickets with a paper but no scope notes, ask one question before proceeding: "Which part of this paper should the ticket cover?" Then wait.

For all other ticket types, you usually have enough to start directly.

### 3. Explore the codebase

Read enough of GLIDE to answer:
1. **Where does it live?** Which module and file to create or modify.
2. **What does it mirror?** Which existing class or pattern to follow.
3. **What does it touch?** Which shared utilities, data structures, or CI implementations it depends on.

Start here:
- `glide/estimators/` — for new estimators; read one or two for naming and structure patterns
- `glide/mean_inference_results/` — result dataclasses returned by estimators
- `glide/utils.py` — shared internal helpers
- `glide/confidence_intervals/` — if confidence interval computation is involved
- `glide/samplers/` — if a sampling strategy is involved
- The files the user pointed to — for refactoring and repository tickets

Mirror rule: `glide/foo/bar.py` always pairs with `tests/unit/foo/test_bar.py`.

### 4. Write the ticket

Pick a filename: lowercase, hyphen-separated, prefixed with the type (`feat-`, `ref-`, `doc-`, `repo-`), derived from the algorithm or topic. Write to `tickets/<filename>.md`.

Load the template for the ticket type from `references/`:

| Type | Template file |
|---|---|
| Feature | `references/template-feature.md` |
| Refactoring | `references/template-refactoring.md` |
| Documentation | `references/template-documentation.md` |
| Repository | `references/template-repository.md` |

Follow the template exactly. Acceptance criteria must be specific to this ticket. Never include items already covered by the team's PR template (`.github/PULL_REQUEST_TEMPLATE.md`) — the following are required on every PR and must not appear in ticket-specific criteria:

- `make lint` passes
- `make type-check` passes
- `make coverage` (100%) passes
- `make doc` builds without warnings
- `CHANGELOG.md` updated

Only write criteria that are unique to this particular ticket: a specific class that must exist, a specific behaviour that must hold, a specific numerical result that must match.

---

All pseudocode must follow the conventions in `CLAUDE.md` (typing, return statements, arrays, naming, docstrings, comments) so developers can copy-adapt without translation.

Do not insert line breaks in the middle of a paragraph; only break between distinct paragraphs or sections or within code or docstrings.