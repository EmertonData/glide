---
name: rename
description: Executes a full renaming task in the GLIDE codebase — renaming a class, function, parameter, module, sampler/estimator/monitor, or concept (e.g. `PPIMeanMonitor` → `EmpiricalPPRM`, `cluster_ppi.py` → `clustered_ppi.py`, `n_total` → `n_samples`). Use whenever the user says "rename X to Y", asks to rename a class/function/parameter/module/file, or describes a renaming ticket from the GitHub Projects board — even for a rename that looks trivial, since a bare find-and-replace reliably misses variable names, plot labels, notebook prose, and doc cross-references.
---

## Overview

- A rename spans more than the source code: the old name resurfaces in places a mechanical search over `.py` files doesn't reach:
  - Local variable names in notebooks.
  - Plot color constants.
  - f-string labels.
  - Markdown headers.
  - Words derived from the same root that aren't the literal renamed token.
- Grepping for the old name is necessary but not sufficient on its own. Full coverage additionally requires:
  - A concrete, ordered checklist.
  - Actually reading each affected file in full rather than trusting grep snippets.
  - A mandatory second pass.
- Follow the steps below in order.

## Step 1 — Pin down every rename before touching code

- If the user's request doesn't fully specify old → new for *every* symbol involved, ask before starting.
- A single conceptual rename is often several literal renames at once — renaming one estimator class can also require renaming its module file, its test file, a related sampler class, and a notebook titled around the old name.
- Identify explicitly, before starting:
  - **Every literal symbol renamed**: class name, function/method name, parameter name, module path, file path (glide source, test file, notebook file).
  - **The root word's surface forms** — for each renamed identifier, how it can appear in prose and code, since every one of these needs its own grep pattern:
    - PascalCase (`ClusterPPI`)
    - snake_case (`cluster_ppi`)
    - space-separated natural language (`Cluster PPI`, `cluster ppi`)
    - any acronym/abbreviation the concept has (e.g. `PPI` vs `PPRM`)
    - plural/derived forms sharing the same root — renaming a `Cluster*` class can also touch unrelated prose like "cluster classical estimator"; audit the whole word family, not just the exact old identifier
  - Whether the rename is a **pure rename** (same behavior, new name) or bundled with other changes, so the CHANGELOG entry doesn't conflate them.
- A rename of a public symbol is a straight rename, not a deprecation: remove the old name outright rather than keeping it as an alias with a deprecation warning. Record the change via the CHANGELOG's `Changed` entry instead.

## Step 2 — Rename the core implementation first

- Rename in this order, since everything else propagates from here:
  1. The class/function/parameter/module itself in `glide/`.
  2. Its export in the relevant `glide/*/__init__.py` (estimators, samplers, monitors, simulators, etc.).
  3. Its docstring's natural-language prose (summary, parameter descriptions, `Notes`/`References` sections) — these describe the symbol in words, not just in the signature, so update them alongside the code.
  4. If a module or test file's name is derived from the symbol, rename the file with `git mv` (not delete+recreate) so blame/history survives.
- Grep before moving on, restricted to `glide/`:

  ```bash
  rg -i '<old_name_surface_forms>' glide/
  ```

- Anything left here means Step 2 isn't done — don't proceed until this is clean.

## Step 3 — Find every affected file, then fan out across them in parallel

- Run a repo-wide, case-insensitive search for every surface form from Step 1 to build the full list of candidate files:

  ```bash
  rg -il '<old_name_surface_form_1>|<old_name_surface_form_2>|...' \
    glide/ tests/ docs/ README.md CONTRIBUTING.md CHANGELOG.md mkdocs.yml
  ```

- This list typically spans:
  - **Unit and functional tests** mirroring the `glide/` structure — rename the test file itself with `git mv` if it's named after the symbol, including mock variable names that reference it.
  - **Docstrings and doctest `Examples` blocks**.
  - **Tutorial and deep-dive/scientific-validation notebooks** — check headers, bold prose, table cells, imports, variable names, plot labels, f-strings, `print()` statements, dict keys used as legend labels, and color constant names *and* their inline comments. Notebook titles in particular lag behind code renames easily since the first cell is easy to skim past.
  - **`docs/api/*.md`** — mkdocstrings `:::` directives and summary tables, both the link anchor and the display text.
  - **`docs/tutorials/index.md`**.
  - **`docs/landing/`** — `tests/generate_fixtures.py` imports directly from `glide.estimators` and `glide.simulators`; a stale import fails `make check-landing-fixtures` (chained into `make tests`).
  - **`mkdocs.yml`** — nav display titles *and* file paths/slugs, which must change if a notebook was renamed.
  - **`README.md`** — the "Implemented Algorithms" table, citation links.
  - **`CONTRIBUTING.md`** — the ASCII architecture directory tree lists module filenames like `├── ppi.py`, easy to miss since it reads as prose, not code.
  - **`CHANGELOG.md`** — add a bullet under `### Changed` in `## [Next release]`, at the top of that section's list; don't touch the Contributors line. If the symbol was introduced by an entry still under `## [Next release]`, rename it in place within that entry instead of adding a second bullet. If the symbol was introduced in an already-released version, leave that historical entry untouched and record the rename as a new bullet under `## [Next release]` instead.
- For every candidate file, actually **read the whole file** before editing it:
  - Never rely on grep's context lines alone — a snippet can hide additional occurrences elsewhere in the file.
  - Some hits need judgment (is this word derived from the renamed root, or a coincidental unrelated use of the same English word?) that only reading the surrounding content resolves.
- Once the core rename from Step 2 is done, every remaining file is independent of the others — none of them need to see each other's edits first. This makes them a good fit for parallel dispatch:
  - Launch one subagent per file (or per small cluster of closely related files, like a notebook and its mirrored test) via the Agent tool, running in parallel rather than one at a time.
  - Give each subagent: the full old → new mapping and the surface-forms list from Step 1, the specific file path to work on, and the guidance above (read the whole file first, use NotebookEdit for `.ipynb` files rather than raw JSON edits, watch for derived words in prose).
  - Since each file's task is a bounded, well-specified rename once the mapping is fixed, a cheaper/faster model (`haiku`) is a good fit for most of these dispatches — reserve a stronger model for any file where the rename is entangled with genuinely ambiguous prose.
  - Have each subagent report back a short summary of what it changed so the changes can be spot-checked.
  - Before moving to Step 4, skim the subagents' changes together for phrasing consistency (e.g., one file settling on "Clustered PPI" while another says "Cluster-level PPI") — Step 4's grep only catches leftover old names, not inconsistent new-name phrasing across files.

## Step 4 — Mandatory second-pass verification

- Do this even if Step 3 felt thorough. Rename tasks in this codebase reliably leave stragglers on the first pass, which is exactly why this step exists as a hard requirement rather than an optional sanity check.
- Re-run the repo-wide grep from Step 3 for every surface form, across the whole tree.
- Read every remaining hit:
  - Expect some to be legitimate (e.g. the old name inside an already-released CHANGELOG entry deliberately left untouched, or a genuinely unrelated word that happens to share a substring) — confirm that deliberately for each hit rather than assuming it's fine.
  - Fix anything unaddressed and re-run the grep.
- Don't stop until a full sweep comes back clean, net of the deliberate exceptions.

## Step 5 — Run the quality gates

```bash
make lint
make type-check
make tests
make coverage
make test-notebooks
```

- `make tests` runs with `--doctest-modules` (catches stale doctest imports/examples) and chains `check-landing-fixtures` (catches a stale import in `docs/landing/tests/generate_fixtures.py`).
- `make coverage` requires 100%: an old alias or unreachable branch left behind shows up as a coverage drop even when `make tests` passes.

## Step 6 — Done, and close the loop for next time

- The rename is ready for review once Steps 1–5 pass. The `create-pr` skill can take it from here to open the PR.
  - Mention to the user that the PR description should note this is a rename (e.g. `Renames X to Y (module_a → module_b), propagating through tests, docs, and tutorials`).
- Before finishing, explicitly ask the user to flag any missed instance of the old name they spot later (in review, in a follow-up commit, or in a future session) — this checklist was itself built from exactly that kind of feedback, and it stays accurate only if new misses get folded back in.
- If the user later reports a missed instance, don't just fix that one occurrence and move on:
  - Treat it as a gap in this skill's checklist, not a one-off mistake.
  - Update Step 3's file-type list (or add a new bullet) so the category is covered going forward.
  - Add a new before/after example to `references/pr-history-examples.md`, following the format of the existing entries, so future runs recognize the same pattern.

See `references/pr-history-examples.md` for concrete before/after snippets of the categories of stray reference described in Step 3 — useful as a sanity check for what "stray" looks like in practice.
