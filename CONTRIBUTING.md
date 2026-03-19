# Contributing to GLIDE

<p align="center">
  <a href="https://github.com/EmertonData/glide/actions/workflows/code_quality.yml"><img src="https://github.com/EmertonData/glide/actions/workflows/code_quality.yml/badge.svg" alt="Code quality"></a>
  <a href="https://codecov.io/gh/EmertonData/glide"><img src="https://codecov.io/gh/EmertonData/glide/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://pypi.org/project/glide-py/"><img src="https://img.shields.io/pypi/v/glide-py" alt="PyPI"></a>
  <a href="https://pypi.org/project/glide-py/"><img src="https://img.shields.io/pypi/pyversions/glide-py" alt="Python versions"></a>
  <a href="https://glide-py.readthedocs.io/en/latest/"><img src="https://app.readthedocs.org/projects/glide-py/badge/?version=stable" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
</p>

Thank you for considering a contribution to GLIDE! This guide covers everything you need to set up your environment, understand the codebase, and submit a pull request.

Depending on what you want to do, jump to the relevant section:

- **Found a bug?** → [Bug fixes](#1-bug-fixes)
- **Want to add a new estimator or feature?** → [New features](#2-new-features)
- **Fixing docs or adding an example?** → [Documentation](#3-documentation)
- **Improving CI, tooling, or the Makefile?** → [Repository hygiene](#4-repository-hygiene)
- **Restructuring code without changing behaviour?** → [Refactoring](#5-refactoring)

Before writing any code, please [open an issue](https://github.com/EmertonData/glide/issues) to discuss the scope of your change. For new statistical methods, a reference to a published paper is required. When you are ready to submit, fork the repository, create a branch off `main`, and open a pull request against `main`.

---

## Setup

GLIDE uses [uv](https://docs.astral.sh/uv/) to manage the virtual environment and all dependency groups.

**1. Install uv** (skip if already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Create the virtual environment and install all dependencies:**

```bash
make venv
# equivalent to: uv sync --all-groups
```

This installs the main package, test dependencies, and documentation dependencies in one step.

**3. Verify the setup by running the test suite:**

```bash
make tests
```

---

## Architectural overview

The package is organised around three concerns: **estimators**, **core building blocks**, and **I/O**.

```
glide/
├── estimators/          # Public API — statistical estimators
│   ├── estimator_protocol.py   # Structural protocol every estimator must satisfy
│   ├── classical.py            # ClassicalMeanEstimator (baseline, no proxy labels)
│   ├── ppi.py                  # PPIMeanEstimator (PPI and PPI++)
│   ├── asi.py                  # ASIMeanEstimator (Active Statistical Inference)
│   └── ...
│
├── core/                # Shared building blocks (not part of the public API)
│   ├── dataset.py              # Dataset container (labeled + unlabeled splits)
│   ├── clt_confidence_interval.py  # CLT-based confidence interval computation
│   ├── simulated.py            # Utilities for generating simulated datasets
│   ├── utils.py                # General-purpose helpers
│   ├── mean_inference_result/  # Result types returned by estimators
│   │   ├── base.py                 # MeanInferenceResult (abstract base)
│   │   ├── classical.py            # ClassicalMeanInferenceResult
│   │   └── semi_supervised.py      # SemiSupervisedMeanInferenceResult
│   └── ...
│
└── io/                  # Serialisation helpers (e.g., to_json)
    └── ...
```

**How the pieces fit together.** Each estimator accepts a `dataset` (a `core.dataset.Dataset`) and produces an inference result object (e.g. a `MeanInferenceResult` subclass). For example, semi-supervised mean estimators (`PPIMeanEstimator`, `ASIMeanEstimator`) return a `SemiSupervisedMeanInferenceResult` that carries both the corrected point estimate and metadata about the used dataset and algorithm; the classical estimator returns a `ClassicalMeanInferenceResult`. Confidence intervals are always computed through `core.clt_confidence_interval`. The `io` module handles serialisation of these result objects.

**Where to make changes:**

| Goal | Where to look |
|---|---|
| New estimator | If your estimator belongs to an already present family, implement it in the existing file, e.g. PPI based methods are in `glide/estimators/ppi.py`. Otherwise, add a file under `glide/estimators/`. Make sure you implement the protocol in `estimator_protocol.py` and export from `glide/estimators/__init__.py` |
| New result type | If needed, add a class under `glide.core` to represent your result type. For example, mean estimation results should subclass `MeanInferenceResult` under `glide/core/mean_inference_result/` |
| Shared maths / utils | `glide/core/` |
| Serialisation | `glide/io/` |

---

## Possible contributions

Contributions are listed below by priority.

### 1. Bug fixes

Reproduce the bug in a failing test first — this confirms the bug exists and guarantees it stays fixed. Then make the minimal code change that makes the test pass.

### 2. New features

Open an issue to discuss scope before writing any code. New estimators should be backed by a scientific publication; include the reference in the issue and in the class docstring.

#### Adding a new estimator — step by step

1. **Identify** the inputs, outputs, and any tunable hyperparameters.
2. **Create** `glide/estimators/<name>.py` if necessary. Implement a class that satisfies `EstimatorProtocol` (defined in `estimator_protocol.py`):
   - `estimate(dataset)` runs the method and returns an inference result object. Use one from `glide/core` (e.g. a `MeanInferenceResult` subclass) or create one that suits your needs.
   - If your estimator has hyperparameters, these should be optional parameters of `estimate()` with default values.
3. **Export** the new class from `glide/estimators/__init__.py`.
4. **Write tests** in `tests/estimators/test_<name>.py`. Cover at minimum:
   - Correct output type and shape.
   - Known analytical results (e.g., the estimator reduces to the classical mean when in special cases).
   - Doctests in the class docstring.
5. **Write a numpy-style docstring** that includes the paper reference, parameter descriptions, and a short `Examples` section with a runnable doctest. See existing estimators for inspiration.
6. **Add an example notebook** under `docs/examples/` demonstrating the estimator on a realistic dataset.
7. **Update `CHANGELOG.md`** under the `[Unreleased]` section.

### 3. Documentation

Corrections, clarifications, and new examples live in `docs/`. Build the docs locally with:

```bash
make doc
# opens a local server at http://127.0.0.1:8000
```

### 4. Repository hygiene

Improvements to CI, Makefile targets, GitHub Actions workflows, or dependency configuration. These changes should not affect the public API or test behaviour.

### 5. Refactoring

Restructuring code without changing observable behaviour. Refactoring PRs must be accompanied by the full passing test suite and must not be bundled with functional changes.

---

## Checklist / Validation process

All quality checks map to `Makefile` targets. Run them before opening a PR:

| Check | Command |
|---|---|
| Linting | `make lint` |
| Type checking | `make type-check` |
| Tests | `make tests` |
| Coverage report | `make coverage` |
| Doc build | `make doc` |

Before opening a PR, all of the following must hold:

- [ ] `make lint` passes (ruff, no errors)
- [ ] `make type-check` passes (ty, no errors)
- [ ] `make tests` passes
- [ ] `make coverage` reports 100% coverage; no lines or branches excluded without justification
- [ ] All new public classes and functions have numpy-style docstrings with a runnable `Examples` / doctest section
- [ ] If documentation was changed, `make doc` completes without warnings and changes are properly rendered
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] PR title and description are clear; new estimators include a paper citation

**A note on LLM-assisted contributions.** LLM usage is welcome and must be disclosed in the PR description. Reviewers should be aware that LLM-generated code tends to increase review burden: it is often verbose, introduces unnecessary abstractions, and may silently diverge from the project's conventions. Contributors are expected to thoroughly read, understand, and validate every line before submitting — not just run the tests. Undisclosed or unvalidated LLM output is grounds for requesting a rewrite.
