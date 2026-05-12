# Documentation ticket template

```markdown
# [Documentation title: what is being written or improved]

## Background

[1–2 paragraphs. What is missing, inconsistent, or unclear in the current documentation. Be specific — point to the relevant files, sections, or docstrings. Explain why this matters for users or contributors (e.g. "the API reference omits the `alpha` parameter", "the user guide for samplers was never written", "three different notebooks use different notation for the same quantity").]

## Design choices

[The proposed approach. What format or structure to use and why. If the content touches maths, specify the notation convention. If it touches code examples, specify which public API to import from. Note any existing sections the new content should align with.]

## Content plan

[What to write, section by section or docstring by docstring. Be specific enough that the developer knows exactly what goes where.]

- `glide/path/to/file.py` — `ClassName.method_name`: [what the docstring should cover]
- `docs/user-guide/topic.md`: [sections to add or revise]
- `notebooks/topic.ipynb`: [cells to add or revise]

## Acceptance criteria

- [ ] [Name the specific artefact completed, e.g. "`IPWPTDMeanEstimator` docstring includes a worked numerical example in the `Examples` section"]
- [ ] [Another specific artefact, e.g. "The sampler user guide covers `StratifiedSampler` with a code snippet showing how to pass strata labels"]
- [ ] [...]
```
