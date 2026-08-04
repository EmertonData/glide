# Contributing to GLIDE

Thank you for considering a contribution to GLIDE! This guide covers everything you need to set up your environment, understand the codebase, and submit a pull request.

Depending on what you want to do, jump to the relevant section:

- **Found a bug?** → [Bug fixes](#1-bug-fixes)
- **Want to add a new estimator or feature?** → [New features](#2-new-features)
- **Fixing docs or adding an example?** → [Documentation](#3-documentation)
- **Improving CI, tooling, or the Makefile?** → [Repository hygiene](#4-repository-hygiene)
- **Restructuring code without changing behaviour?** → [Refactoring](#5-refactoring)
- **New to git, or don't know how to code?** → [A note for non-technical contributors](#a-note-for-non-technical-contributors)

Before writing any code, please [open an issue](https://github.com/EmertonData/glide/issues) to discuss the scope of your change. This is highly recommended and especially important for new estimators and samplers: sharing the reference paper upfront gives maintainers a chance to read it and frame the ticket to guide your implementation. Once the scope is agreed, follow the [Contribution workflow](#contribution-workflow) below to submit your change. The PR template lists all conditions that must be satisfied before requesting a review.

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
```

This installs the main package, test dependencies, and documentation dependencies in one step.

**3. Verify the setup by running the test suite:**

```bash
make tests
```

**4. Install the git pre-commit hooks:**

GLIDE uses [prek](https://github.com/rec/prek) (a lightweight pre-commit hook runner) configured in `prek.toml`. The hooks run automatically on every `git commit` and enforce formatting, type checking, and notebook output stripping.

Install the hooks once after cloning:

```bash
uv run prek install
```

**5. Testing notebooks locally (optional):**

The project includes example notebooks in `docs/`. To test all notebooks locally:

```bash
make test-notebooks
```

Note: Notebook testing also runs in CI for all pull requests, so local testing is optional. The CI workflow ensures notebooks are executed and validated before merge.

---

## Contribution workflow

Once an issue is open and its scope agreed, follow these steps to submit your change.

**1. Sync with `main`**

```bash
git checkout main
git pull origin main
make venv
```

**2. Create a branch**

```bash
make branch name=<your-branch-name>
```

This checks out `main`, pulls the latest changes, and creates your branch in one step. Name branches with the same prefixes as commits (see below), e.g. `feat/user-auth`, `fix/bug-fix`, `doc/improve-rendering`.

**3. Make your changes and commit**

Before pushing, run the checks that CI will run:

```bash
make lint
make type-check
make coverage
```

Write commit messages using [Conventional Commits](https://www.conventionalcommits.org/) prefixes: `feat:`, `fix:`, `doc:`, `ref:`, `test:`, `chore:`, `perf:`, `style:`, `build:`, `ci:`.

**4. Open a pull request**

Push your branch and open a pull request against `main`, linking the related issue and adding the relevant labels. Fill in the PR template: it lists all conditions that must be satisfied before requesting a review.

If you use Claude Code, you don't have to do any of this by hand: it has a `create-pr` skill that pushes the branch, fills in the template from your diff, updates the changelog, and applies labels. Just ask it to open the PR for you.

**5. Code review**

A maintainer reviews the PR for logic, design, and quality. Address every comment, either with a follow-up commit or a reply. The reviewer resolves each conversation once it is addressed.

**6. Squash and merge**

Once approved, the PR is squashed into a single, clean commit on `main`.

### A note for non-technical contributors

You do not need to know how to code to contribute to GLIDE. If you have a notebook, a script, or just an idea for a fix or a new example, open [Claude Code](https://claude.com/product/claude-code) in a clone of this repository and ask it to read this file and open a pull request for you. Claude Code can run every command above (branching, testing, committing, opening the PR) on your behalf, so a working knowledge of Python or git is not a prerequisite. You are still responsible for reviewing what it produces before requesting a review, per the [note on LLM-assisted contributions](#a-note-on-llm-assisted-contributions) below.

---

## Architectural overview

The package is organised around five concerns: **estimators**, **samplers**, **monitors**, **core building blocks**, and **I/O**.

```
glide/
├── estimators/             # Public API — mean estimators
│   ├── ppi.py
│   ├── ...
│
├── monitors/               # Public API — drift monitors over batched data
│   ├── asymptotic_pprm.py
│   ├── ...
│
├── engines/                # Internal — per-unit body shared by estimators and monitors
│   ├── ppi.py
│   ├── ...
│
├── samplers/               # Public API — sampling strategies
│   ├── active.py
│   ├── ...
│
├── simulators/               # Public API — synthetic data generators for tests
│   ├── binary.py
│   ├── ...
│
├── confidence_intervals/   # Confidence interval
│   ├── base.py
│   ├── ...
│
├── confidence_sequences/   # Anytime-valid confidence sequences (used by monitors)
│   ├── base.py
│   ├── ...
│
├── mean_inference_results/ # Result types returned by estimators
│   ├── base.py
│   ├── ...
│
├── mean_monitoring_results/ # Result types returned by monitors
│   ├── base.py
│   ├── ...
│
├── core/                   # Centralized parameter validation helpers
│   └── validation.py
│
├── utils.py                # General-purpose helpers
│
└── io/                     # Serialisation helpers (e.g., to_json)
    └── export.py
```

**How the pieces fit together.** Estimators accept raw NumPy arrays and return a `MeanInferenceResult` subclass: prediction-powered estimators return a `PredictionPoweredMeanInferenceResult`, classical ones a `ClassicalMeanInferenceResult`. Every result embeds a `ConfidenceInterval` (e.g. `CLTConfidenceInterval`). Samplers produce the labeled arrays that estimators consume. Monitors follow the same shape for batched, accumulating data: they return a `MeanMonitoringResult` subclass embedding a `ConfidenceSequence` (e.g. `AsymptoticConfidenceSequence`). The `io` module serialises result objects. Both estimator and monitor shells delegate their per-unit body (validate, transform, tune, then compute mean and std) to a shared `Engine` from `glide/engines/`: the estimator shell wraps one call over the whole dataset, the monitor shell loops it once per batch, and each shell only packages the result.

---

## Possible contributions

Contributions are listed below.

### 1. Bug fixes

Reproduce the bug in a failing test first — this confirms the bug exists and guarantees it stays fixed. Then make the minimal code change that makes the test pass.

### 2. New features

New estimators and samplers should be backed by a scientific publication. Please first [open an issue](https://github.com/EmertonData/glide/issues) sharing the reference paper to give maintainers a chance to read and frame it to guide your implementation.

**Adding a new estimator — step by step**

1. **Identify** the inputs, outputs, and any tunable hyperparameters.
2. **Implement the engine** that owns the per-unit body (validate, transform, tune, then compute mean and std):
   - Create `glide/engines/<name>.py` with a class exposing `prepare(...)`, `fit_tuning_parameter(dataset, power_tuning)`, and `compute_mean_and_std(dataset, tuning_parameter)`. `prepare` takes the family's explicit named inputs and returns the validated, transformed dataset (a plain tuple of arrays, or a single array, whichever fits the family — no bespoke dataclass needed).
   - This engine will be reused unchanged by a monitor over the same family later on; keep it free of anything estimator-specific (confidence intervals, result packaging).
3. **Implement the estimator shell**:
   - Create `glide/estimators/<name>.py`. In `__init__`, instantiate the engine (and, if you need an effective-sample-size baseline, the paired classical engine) as instance attributes.
   - `estimate(array1, array2, ...)` calls `prepare` → `fit_tuning_parameter` → `compute_mean_and_std` on the engine, builds a `ConfidenceInterval`, and packages an inference result. Reuse a result type from `glide/mean_inference_results` (e.g. a `MeanInferenceResult` subclass) or add a new one there.
   - If your estimator has hyperparameters, these should be optional parameters of `estimate()` with default values.
4. **Export** the new class from `glide/estimators/__init__.py`.
5. **Write unit tests**: engine-level tests in `tests/unit/engines/test_<name>.py` covering validation, tuning, and mean/std against hardcoded expected values; shell-level tests in `tests/unit/estimators/test_<name>.py` covering:
   - Correct output type and shape.
   - Known outputs for fixed inputs.
   - Doctests in the class docstring.
6. **Write functional tests** in `tests/functional/estimators/test_<name>.py`. If applicable, test expected behaviors and properties of your estimator in specific situations (e.g., the estimator reduces to the classical mean in special cases), see existing files in `tests/functional/estimators` for examples
7. **Write a numpy-style docstring** that includes the reference paper, parameter descriptions, and a small `Examples` section with a minimalistic runnable doctest. See existing estimators for inspiration.
8. **Add an example script** in `docs/examples/plot_<name>.py` demonstrating the estimator on some synthetic data.
9. **Update `CHANGELOG.md`** under the `[Next release]` section.

Exact numerics matter here: the estimator doctest's printed values are the primary guard that the engine reproduces today's math unchanged.

**Adding a new sampler — step by step**

1. **Identify** the inputs the sampler requires (e.g. proxy labels, uncertainty scores, stratum labels), the budget parameter, and what values it returns.
2. **Implement** the sampler class:
   - Create `glide/samplers/<name>.py`.
   - `sample(...)` runs the sampling procedure and returns the computed values (at least a vector `xi` of sampling indicators and possibly a vector `pi` of sampling probabilities).
   - If your sampler has hyperparameters, these should be optional parameters of `sample()` with default values.
3. **Export** the new class from `glide/samplers/__init__.py`.
4. **Write unit tests** in `tests/unit/samplers/test_<name>.py`. Cover at minimum:
   - Correct output type and shape.
   - Known analytical results (e.g., uniform inputs should yield equal probabilities).
   - Edge cases for input parameters (e.g. budget equals dataset size).
   - Doctests in the class docstring.
5. **Write functional tests** in `tests/functional/samplers/test_<name>.py`. If applicable, test expected behaviors and properties of your sampler. See existing files in `tests/functional/samplers` for examples.
6. **Write a numpy-style docstring** that includes the reference paper, parameter descriptions, and a small `Examples` section with a minimalistic runnable doctest. See existing samplers for inspiration.
7. **Update `CHANGELOG.md`** under the `[Next release]` section.

**Adding a new monitor — step by step**

1. **Identify** the inputs (e.g. batched labels, the alarm threshold), outputs, and any tunable hyperparameters.
2. **Reuse the family's engine.** If an estimator for this family already exists, its engine in `glide/engines/` already implements the per-batch body (validate, transform, tune, then compute mean and std); a monitor never re-derives this math, it only loops the engine once per batch. If no estimator exists yet for this family, implement the engine first (see "Adding a new estimator" above).
3. **Implement a thin façade**:
   - Create a properly named file `glide/monitors/<name>.py` for your monitor, as a class inheriting `AsymptoticRM` (from `glide/monitors/base.py`), parameterized by the engine's dataset type, with an `engine` class attribute set to an instance of the family's engine.
   - `detect(y, batches, ...)` is the family's explicit public signature: it assembles the per-sample arrays into `fields`, calls the shared `_detect`, and packages the result inline from what `_detect` returns (no separate result-builder method). Reuse a result type from `glide/mean_monitoring_results` (e.g. a `MeanMonitoringResult` subclass) or add a new one there.
   - The result embeds a `ConfidenceSequence`: reuse one from `glide/confidence_sequences` (e.g. `AsymptoticConfidenceSequence`) or add a new one there.
   - If your monitor has hyperparameters, these should be optional parameters of `detect()` with default values.
   - The monitor is selected by which class you instantiate, never inferred from the shape of the input arrays.
4. **Export** the new class from `glide/monitors/__init__.py`.
5. **Write unit tests** in `tests/unit/monitors/test_<name>.py`. Cover at minimum:
   - Correct output type and shape.
   - Known outputs for fixed inputs.
   - Doctests in the class docstring.
6. **Write functional tests** in `tests/functional/monitors/test_<name>.py`. If applicable, test expected behaviors and properties of your monitor in specific situations (e.g., the monitor's per-batch estimates match the corresponding one-shot estimator), see existing files in `tests/functional/monitors` for examples.
7. **Write a numpy-style docstring** that includes the reference paper, parameter descriptions, and a small `Examples` section with a minimalistic runnable doctest. See existing monitors for inspiration.
8. **Update `CHANGELOG.md`** under the `[Next release]` section.

**Adding a new estimator family end-to-end.** A family's three pieces share one engine: the `Engine` in `glide/engines/` owns the per-unit body once; the estimator shell calls it a single time over the whole dataset; the monitor façade calls it once per batch (tuning fitted on the batches preceding each one). Write the engine first, then the estimator shell, then the monitor façade on top of it, in that order — none of the three re-derives the others' math. The engine is always selected by which estimator or monitor class you instantiate, never inferred from the input arrays' shape.

### 3. Documentation

Corrections, clarifications, and new examples live in `docs/`. Build the docs locally with:

```bash
make doc
```

### 4. Repository hygiene

Improvements to CI, Makefile targets, GitHub Actions workflows, or dependency configuration. These changes should not affect the public API or test behaviour.

### 5. Refactoring

Restructuring code without changing observable behaviour. Refactoring PRs must be accompanied by the full passing test suite and must not be bundled with functional changes.

---

## A note on LLM-assisted contributions

LLM usage is welcome and must be disclosed in the PR description. Reviewers should be aware that LLM-generated code tends to increase review burden: it is often verbose, introduces unnecessary abstractions, and may silently diverge from the project's conventions. Contributors are expected to thoroughly read, understand, and validate every line before submitting — not just run the tests. Undisclosed or unvalidated LLM output is grounds for requesting a rewrite.
