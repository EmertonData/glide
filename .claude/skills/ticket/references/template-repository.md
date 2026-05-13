# Repository ticket template

```markdown
# [Repository title: what is being improved and why]

## Background

[1–2 paragraphs. What is the current problem or gap: a missing CI check, a slow test suite, a flaky pipeline, an outdated dependency, missing pre-commit hooks. Be concrete — link to the relevant config files, CI runs, or failing steps. Explain the impact (e.g. "coverage is not enforced on PRs", "the type-checker runs but its output is ignored", "notebook outputs are committed to the repository").]

## Design choices

[The proposed approach. What tool, configuration, or workflow to adopt and why. If there are alternatives, explain the trade-off briefly.]

## Task breakdown

- [ ] [Specific subtask — name the file to create or modify using full, self-explaining names (e.g. `compatibility.yml` not `compat.yml`, `spec0-matrix.yml` not `spec0.yml`)]
- [ ] [Another subtask]
- [ ] [...]

## Acceptance criteria

- [ ] [Specific and testable, e.g. "A PR that introduces a new public function without a docstring is blocked by CI"]
- [ ] [Another specific deliverable, e.g. "Running `make pre-commit` strips all notebooks from outputs before committing"]
- [ ] [...]
```
