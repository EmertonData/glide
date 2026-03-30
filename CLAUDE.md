# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GLIDE** (Generated Label Inference & Debiasing Engine) is a Python scientific library for rigorous evaluation of GenAI systems using hybrid human/proxy annotations. It implements semi-supervised statistical inference methods (PPI, ASI, Classical) that combine small labeled datasets with large proxy-labeled datasets to produce valid, debiased statistical estimates.

- Package name on PyPI: `glide-py`
- Python 3.12+, managed with `uv`
- New features must reference a scientific publication

## Development Commands

```bash
make venv          # Create virtual environment (uv sync --all-groups)
make tests         # Run tests: uv run pytest . -vsx
make coverage      # Full coverage report (100% required)
make lint          # Ruff linting
make type-check    # Type checking with ty
make pre-commit    # Run pre-commit hooks (ruff + nbstripout + ty)
make build         # Build distribution
make doc           # Serve docs locally (mkdocs)
```

Run a single test file: `uv run pytest tests/unit/test_foo.py -vsx`

## Architecture

The package has three layers:

**`glide/estimators/`** — Public API. Each estimator implements the `MeanEstimator` Protocol (`estimate(dataset, **fields) -> MeanInferenceResult`). Current estimators: `ClassicalMeanEstimator`, `PPIMeanEstimator`, `ASIMeanEstimator`.

**`glide/core/`** — Internal building blocks:
- `dataset.py` — `Dataset` extends `list` with column/record access
- `clt_confidence_interval.py` — CLT-based confidence intervals
- `mean_inference_result/` — Result dataclasses (`ClassicalMeanInferenceResult`, `SemiSupervisedMeanInferenceResult`)
- `simulated_datasets.py` — Test data generators

**`glide/io/`** — Serialisation helpers (JSON export).

## Git Workflow

```bash
git checkout main && git pull origin main && make venv  # Step 0: sync
git checkout -b feat/my-feature                         # Step 1: branch
# ... commit with conventional messages: feat:, fix:, doc:, ref:
# Step 3: open PR on GitHub, link to related issue
# Step 4: address review comments, push new commits
# Step 5: squash and merge into main
```

Branch naming: `feat/`, `fix/`, `doc/`, `ref/` prefixes.

## Definition of Done

Every PR must satisfy all of the following before merge:

- [ ] `make lint` passes
- [ ] `make type-check` passes
- [ ] 100% coverage on unit tests (`make coverage`)
- [ ] NumPy-style docstrings on all public API
- [ ] `make doc` builds without warnings, new docstrings rendered in API section
- [ ] API reference section updated if user-facing
- [ ] AI code review
- [ ] Human code review
- [ ] Lead tech review

## Testing Requirements

- Tests live in `tests/unit/` and `tests/functional/`
- pytest runs with `--import-mode=importlib --doctest-modules`, so module docstrings are also tested
- Every new feature needs: doctests in the docstring + unit tests + analytical verification (compare against known closed-form results)
- 100% coverage is enforced; exempted files: `estimator_protocol.py`, `__init__.py` files

## Code Conventions

- Line length: 120 (configured in ruff)
- NumPy-style docstrings with paper references for all public API
- Public API docstrings must include an `Examples` section with runnable doctests
- Method parameters for Dataset field names use `*_field` suffix (e.g., `label_field`, `score_field`)

### Naming

- **Self-explanatory, no abbreviations** — BAD: `ess`, GOOD: `effective_sample_size`
- **Consistent across classes** — all estimators must use the same name for equivalent concepts (e.g., don't use `ppi_mean` in one and `asi_mean` in another; use `compute_mean_estimate` everywhere)

### No Redundancy

Information should appear exactly once. If two classes share logic (e.g., a validation raise), extract it rather than duplicating. If two doc sections say the same thing, remove one.

## Adding a New Estimator

1. Implement the class in a dedicated module under `glide/estimators/`, following the structure of `PPIMeanEstimator`
2. Mirror the test structure of `tests/unit/test_ppi.py`
3. Write a scientific validation notebook in `docs/science/`
4. Write a user guide page describing the theory in `docs/user_guide/`
5. Write a tutorial notebook with a business case scenario in `docs/tutorials/`
6. Update the README scientific literature mapping

## Documentation

- MkDocs with mkdocstrings; docs must build without warnings
- Update `CHANGELOG.md` for any user-facing changes (Keep a Changelog format, SemVer)
