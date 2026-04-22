# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GLIDE** (Generated Label Inference & Debiasing Engine) is a Python scientific library for rigorous evaluation of GenAI systems using hybrid human/proxy annotations. It implements prediction-powered statistical inference methods (PPI, ASI, Classical) that combine a small set of labeled data with a large set of proxy-labeled data to produce valid, debiased statistical estimates.

- Package name on PyPI: `glide-py`
- Python 3.12+, managed with `uv`

## Development Commands

```bash
make venv          # Create virtual environment (uv sync --all-groups)
make lint          # Ruff linting
make type-check    # Type checking with ty
make tests         # Run tests: uv run pytest . -vsx
make coverage      # Full coverage report (100% required)
make pre-commit    # Run pre-commit hooks (ruff + nbstripout + ty)
make test-notebooks # Test all Jupyter notebooks
```

Run a single test file: `uv run pytest tests/unit/test_foo.py -vsx`

## Architecture

The package has multiple layers:

**`glide/estimators/`** — Public API. Statistical estimators (classical, prediction-powered, stratified variants).

**`glide/confidence_intervals/`** — Confidence interval implementations depending on statistical methods.

**`glide/samplers/`** — Sampler object implementations for various strategies.

**`glide/core/`** — Internal building blocks: data structures, result dataclasses, test data generators, and shared utilities.

**`glide/io/`** — Serialisation helpers.

## Git Workflow

Branch naming: `feat/`, `fix/`, `doc/`, `ref/` prefixes.

## Definition of Done

Every PR must satisfy all of the following before merge:

- [ ] `make lint` passes
- [ ] `make type-check` passes
- [ ] 100% coverage on unit tests (`make coverage`)
- [ ] NumPy-style docstrings on all public API
- [ ] `make doc` builds without warnings, new docstrings rendered in API section
- [ ] API reference section updated if user-facing

## Testing Requirements

- Tests live in `tests/unit/` and `tests/functional/`
- `tests/unit/` mirrors the `glide/` folder structure exactly (e.g., `glide/core/foo.py` → `tests/unit/core/test_foo.py`)
- pytest runs with `--import-mode=importlib --doctest-modules`, so module docstrings are also tested
- Every new feature needs: doctests in the docstring + unit tests + analytical verification (compare against known closed-form results)
- 100% coverage is enforced
- Test names: `test_<name_of_tested_function>` with optional descriptive suffixes (e.g., `test_generate_binary_dataset_invalid_correlation`)
- One test per distinct function call — do not write redundant tests
- Use the smallest arrays possibles (typically 2 elements, rarely more than 10); tests must be lightning fast
- Use fixtures to factorize pervasive test elements (shared arrays, estimator instances, etc.)
- Existing test files are the canonical reference for structure and patterns — follow `test_ppi.py`, `tests/unit/simulators/test_binary.py`, etc. when writing new test files
- Use `pytest.approx(value, abs=0.01)` when comparing scalar floats in tests
- Use `np.testing.assert_allclose` when comparing arrays of floats in tests
- Use `np.testing.assert_array_equal` when comparing arrays of strings or categories in tests

## Code Conventions

- Line length: 120 (configured in ruff)
- Use `np.hstack` or `np.vstack` instead of `np.concatenate` whenever possible
- Use NumPy vectorization instead of Python for loops whenever possible

### Naming

- **Self-explanatory, no abbreviations** — BAD: `ess`, GOOD: `effective_sample_size`
- **Consistent across classes** — all estimators must use the same name for equivalent concepts (e.g., don't use `ppi_mean` in one and `asi_mean` in another; use `compute_mean_estimate` everywhere)

### Return Statements

Never compute in a `return` statement. Assign the result to a named variable first, then return it.

```python
# BAD
return a * b + c

# GOOD
result = a * b + c
return result
```

### Type Annotations

Use `typing` module generics — never use PEP 604 / PEP 585 built-in aliases:

- BAD: `int | None`, `list[str]`, `dict[str, int]`, `tuple[int, ...]`
- GOOD: `Optional[int]`, `List[str]`, `Dict[str, int]`, `Tuple[int, ...]`

Always import from `typing`: `from typing import Dict, List, Optional, Tuple`.

### Type Conversions

Do not use needless type conversions like `float()` or `int()` unless required by the caller or for debugging purposes.

### Docstrings

- NumPy-style docstrings with paper references for all public API
- Public API docstrings must include an `Examples` section with runnable doctests
- **No docstrings for test functions or private methods** — test code is self-documenting via clear variable names and assertion structure; private methods are internal only.

### Documentation

- MkDocs with mkdocstrings; docs must build without warnings
- Update `CHANGELOG.md` for any user-facing changes (Keep a Changelog format, SemVer)
- Always add elements to the "Added" or "Changed" sections of `CHANGELOG.md` at the top of the existing list
- Avoid using and escaping underscores in math mode in jupyter notebooks.
- Avoid making excessive use of dashes like this — when writing documentation and notebooks. Prefer commas, colons and parentheses where possible.

### No Redundancy

Information should appear exactly once. If two classes share logic (e.g., a validation raise), extract it rather than duplicating. If two doc sections say the same thing, remove one.
