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
make tests         # Run all tests (unit + functional)
make unit-tests    # Run unit tests only
make functional-tests # Run functional tests only
make coverage      # Coverage report on unit tests only (100% required; functional tests are excluded)
make pre-commit    # Run pre-commit hooks via prek (not the standard pre-commit CLI)
make test-notebooks # Test all Jupyter notebooks
make doc            # Build and serve documentation locally with MkDocs
```

Run a single test file: `uv run pytest tests/unit/test_foo.py -vsx`

## Architecture

The package has multiple layers:

**`glide/estimators/`** — Public API. Statistical estimators (classical, PPI, ASI, IPW, PTD, and their stratified variants). Core computation modules (`ppi_core.py`, `ptd_core.py`, `stratified_core.py`) hold shared internals.

**`glide/confidence_intervals/`** — Confidence interval implementations depending on statistical methods.

**`glide/samplers/`** — Sampler object implementations (uniform, stratified, cost-optimal, cost-optimal-random, active).

**`glide/mean_inference_results/`** — Result dataclasses returned by estimators. `base.py` defines `MeanInferenceResult`; `classical.py` and `prediction_powered.py` extend it.

**`glide/core/`** — Centralized parameter validation helpers (`validation.py`). Import from here rather than duplicating checks across modules.

**`glide/scientific_validation.py`** — `run_monte_carlo()` utility used by deep-dive validation notebooks.

**`glide/simulators/`** — Synthetic dataset generators and annotation simulators for testing and validation.

**`glide/io/`** — Serialisation helpers.

## Testing Requirements

- Tests live in `tests/unit/` and `tests/functional/`
- Both directories mirror the `glide/` folder structure exactly (e.g., `glide/estimators/foo.py` → `tests/unit/estimators/test_foo.py`); `tests/functional/` contains tests verifying statistical properties of estimators, samplers and simulators
- pytest runs with `--import-mode=importlib --doctest-modules`, so module docstrings are also tested
- Every new feature needs: doctests in the docstring + unit tests + analytical verification where relevant (assert against hardcoded expected values, not values computed inline by the test itself — e.g., `CLTConfidenceInterval(mean=0, std=1, confidence_level=0.95)` → bounds `±1.96`, or StratifiedPPI with one stratum must equal PPI)
- 100% coverage is enforced
- Test names: `test_<name_of_tested_function>` with optional descriptive suffixes (e.g., `test_generate_binary_dataset_invalid_correlation`)
- Each distinct scenario (input combination, edge case, or error condition) gets its own test function — do not write two test functions that exercise the exact same code path with equivalent inputs
- Use the smallest arrays possibles (typically 2 elements, rarely more than 10); tests must be lightning fast
- Use fixtures to factorize pervasive test elements (shared arrays, estimator instances, etc.)
- Existing test files are the canonical reference for structure and patterns — follow `tests/unit/estimators/test_ppi.py`, `tests/unit/simulators/test_binary.py`, etc. when writing new test files
- Use `pytest.approx(value, abs=<tol>)` when comparing scalar floats in tests — tolerance should be as small as possible given the precision of the expected value
- Use `np.testing.assert_allclose` when comparing arrays of floats in tests
- Use `np.testing.assert_array_equal` when comparing arrays of strings or categories in tests
- No comments within unit test functions. Encode non-obvious derivations as named variables in the test body instead
- Never write a test that doesn't actually test anything — e.g., asserting on a copy made after the function call, or asserting a value that is always true regardless of the implementation
- To assert a mock called once with scalar/list arguments, use `assert_called_once_with(...)`.
- To assert a mock called once with numpy array arguments, use `assert_called_once()` followed by `np.testing.assert_array_equal(mock.call_args[0][i], ...)` for each array argument.
- To assert a mock called multiple times with scalar arguments, use `assert_has_calls([call(...), call(...)])` to verify each call in order.
- To assert a mock called multiple times with numpy arguments, verify each call individually via `call_args_list`.
- Never use `assert_called_with`, `assert_any_call`, or `assert mock.call_count == N` alone.
- Name every mock variable `mock_<mocked_function_name>`, stripping the leading underscore from private functions — BAD: `mock`, `mock_nan`, `mock_bounds`; GOOD: `mock_validate_has_no_nan`, `mock_validate_bounds`.

## Code Conventions

- Line length: 120 (configured in ruff)
- Use `np.hstack` or `np.vstack` instead of `np.concatenate` whenever possible
- Use NumPy vectorization instead of Python for loops whenever possible

### Naming

- **Self-explanatory, no abbreviations** — BAD: `ess`, GOOD: `effective_sample_size`
- **Consistent across classes** — all estimators must use the same name for equivalent concepts (e.g., don't use `ppi_mean` in one and `asi_mean` in another; use `compute_mean_estimate` everywhere)
- **UPPER_CASE for global constants and parameters, lower_case for computed values** — a variable holding a user-supplied budget is `BUDGET`; a variable holding a computed result is `cost`
- **Leading underscore for private or internal helpers** — e.g., `_preprocess`, `_compute_weights`
- **Avoid confusable names** — e.g., `pi_value` reads as `p-value`; choose an unambiguous alternative

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

Never use `object` or `Any` as type annotations, use precise types or protocols instead.

### Type Conversions

Do not use needless type conversions like `float()` or `int()` unless required by the caller or for debugging purposes.

### Error Handling

- Raise `ValueError` for invalid inputs (wrong type, out-of-range value, shape mismatch, NaN where not allowed).
- Raise `RuntimeError` for invalid state — e.g., calling a method before a required prerequisite (like `fit()` before `sample()`).
- Never define custom exception classes.
- Always include the bad value in the message: `f"'param' must be > 0; got {value!r}."` Terminate messages with a period.
- Only validate at public API boundaries. Centralizing checks into a private `_preprocess` helper is fine; re-checking the same condition in downstream private helpers is not.

### Comments

Only add a comment when the *why* is non-obvious: a hidden constraint, a subtle invariant, a workaround for a specific bug, behavior that would surprise a reader. No comments restating what the code does.

### Docstrings

- NumPy-style docstrings for all public API; paper references go exclusively in the `References` section. Never mention papers inline in parameter descriptions, summaries, or notes
- Public API docstrings must include an `Examples` section with runnable doctests
- In doctests, always import from the public package namespaces: `from glide.estimators import ...`, `from glide.samplers import ...`, `from glide.simulators import ...` — never from submodules directly.
- **No docstrings for test functions or private methods** — test code is self-documenting via clear variable names and assertion structure; private methods are internal only.

### Documentation

- MkDocs with mkdocstrings; docs must build without warnings
- Always specify image widths as a percentage (e.g. `width="80%"`), never as fixed pixels.
- Update `CHANGELOG.md` for any user-facing changes (Keep a Changelog format, SemVer)
- Always add elements to the "Added" or "Changed" sections of `CHANGELOG.md` at the top of the existing list
- Keep the `CHANGELOG.md` user-friendly and concise
- Avoid using and escaping underscores in math mode in jupyter notebooks.
- Avoid making excessive use of dashes like this — when writing documentation and notebooks. Prefer commas, colons and parentheses where possible.
- In documentation and tutorials, always spell out "confidence interval" instead of using "CI", which is easily confused with "continuous integration".

#### Tutorials vs. user guides vs. deep dives

These three document types have distinct purposes and must not bleed into each other.

- **Tutorials** show how to use the public API. They must not expose private attributes or methods, must not be math-heavy (that belongs in the user guide), and must tell a coherent end-to-end story. If a tutorial section feels like it is documenting the implementation rather than teaching usage, move it or cut it.
- **Deep dives / validation notebooks** are for scientific validation. They test the statistical validity of one or more workflows, where each workflow covers a full pipeline: data generation, annotation, then estimation. Results are stochastic and depend on a random seed. Monte Carlo simulations must use a single external loop over seeds, calling each workflow once per iteration. Note that workflows do not fully isolate from each other: data generation is typically shared across them. Figures must appear at the right point in the narrative.
- **User guides** document the mathematical foundations of each implemented method (sampling or estimation). For each method, lay out the mathematical setting: expected inputs, computed quantities, and their main properties, taking inspiration from pre-existing sections in the guide. When a reference paper exists, formulas must be consistent with it; notation may diverge from the paper when needed to preserve consistency across the guide as a whole. Never reference the implementation directly: no pseudocode, no mentions of class or method names.

#### Scientific consistency in notebooks

Figures, printed values, and inline text must be mutually consistent — if the text says one number and the figure or table shows another, all occurrences must be corrected together. Sampling rules and notation must match the paper (e.g., πᵢ ∈ (0,1), not [0,1], in theory sections). Statistical framing must be precise: never say "removes bias" without specifying which bias.

### Code Hygiene

Remove debug cells, dead code, and stray blank lines before merging. This applies to Python files and Jupyter notebooks alike.

### No Redundancy

Information should appear exactly once. If two classes share logic (e.g., a validation raise), extract it rather than duplicating. If two doc sections say the same thing, remove one.

### Consistency and Propagation

Before adding a new class, function, or module, find the nearest equivalent already in the codebase and verify that the new code mirrors it in structure, naming, and block ordering. When a good pattern appears in new code that the existing code lacks, propagate it to the existing code in the same PR. When a rename or refactor touches one site, grep for all other sites that use the old name and update them before considering the work done.
