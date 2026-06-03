# Refactoring ticket template

```markdown
# [Refactoring title: what is changing and why]

## TODO

- [ ] <!-- placeholder for refinement actions: update, split, convert to issue, etc. -->

## Background

[1–2 tight paragraphs. Name the specific files, classes, and methods involved. State the current problem (inconsistency, duplication, naming confusion) and the goal state in one sentence each. Skip context that the Implementation section already makes obvious.]

## Design choices

[1 paragraph max, strictly synthetic. What gets renamed, moved, merged, or split, and the key reason for each choice. One sentence per decision is enough — "chose X over Y because Z".

**On extractions:** when shared logic is pulled into a new function or module, callers should call it directly — not wrap it in a one-liner delegate method. One-liner delegates duplicate test surface and add indirection without any benefit. Remove the original method and update every caller to use the new function.]

## Implementation

[Show what changes. For renames, show the mapping. For interface changes, show old and new signatures side by side. For structural changes, show the before and after. The goal is for a developer to understand exactly what to do without needing to interpret anything.

**On tests:** do not list individual test function signatures. Just note what moves (which existing tests are relocated and where they land) and any genuinely new scenario that didn't exist before. Existing tests that already cover the behaviour don't need to be re-described.]

**Files affected:**

| File | What changes |
|---|---|
| `glide/path/to/file.py` | [description] |
| `tests/unit/path/to/test_file.py` | [description] |

**Before / After (where helpful):**

```python
# Before
def old_method_name(self, x: OldType) -> OldReturn:
    ...

# After
def new_method_name(self, x: NewType) -> NewReturn:
    ...
```

## Corner cases

- [Things that could break during the refactor — callers that may not be obvious, implicit contracts, test fixtures that need updating, public API changes that affect users]
- [Another scenario]

## Acceptance criteria

- [ ] [Specific deliverable, e.g. "All estimators expose estimate rather than a class-specific alias"]
- [ ] [Another specific deliverable — what is observable once the refactor is complete]
- [ ] [...]

Do not include items already in the PR template: `make lint`, `make type-check`, `make coverage`, `make doc`, and `CHANGELOG.md` are always required and belong there, not here.
```
