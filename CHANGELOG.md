# Changelog 📋

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### ✨ Added
- ...

---

## [0.2.0] – 2026-03-19

### ✨ Added
- Added `ASIMeanEstimator` for Active Statistical Inference-based mean estimation
- Added `ClassicalMeanEstimator` as a standard baseline estimator
- Added power-tuning support (`λ`) to `PPIMeanEstimator` for narrower confidence intervals
- Added `to_json()` method on inference result objects for easy serialisation
- Added a Quickstart guide to the documentation

### 🔄 Changed
- Refactored `InferenceResult` into dedicated result objects with clearer structure
- Updated release workflow to document version tagging and environment settings explicitly
- Switched documentation layout to full-page width for better readability

### 🐛 Fixed
- Fixed broken badge links in the README

### 💛 Contributors
Thank you to everyone who contributed to this release: @gmartinon-ed, @gherouville-ed, @imerad

---

## [0.1.1] – 2026-03-12

### ✨ Added
- Implemented `PPIMeanEstimator` — the core Prediction-Powered Inference estimator for mean estimation
- Added confidence interval computation on PPI estimates
- Added synthetic dataset generators, including binary data generation, for testing and examples
- Set up MkDocs-based documentation with Read the Docs deployment
- Added docstring examples and doctests for `PPIMeanEstimator`
- Added scientific validation experiments in the documnetation to verify estimator correctness
- Added a code-quality CI workflow covering linting, type checking, and coverage reporting
- Set up CI/CD pipelines for automated TestPyPI and PyPI publishing
- Added Apache 2.0 licence

### 🔄 Changed
- Simplified and harmonized internal variable naming for labelled/unlabelled data splits

### 💛 Contributors
Thank you to everyone who contributed to this release: @gmartinon-ed, @gherouville-ed, @mraki-ed, @vwoelffel-ed, @imerad
