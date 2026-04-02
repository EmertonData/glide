# Changelog 📋

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Next release]

### ✨ Added
- Added tutorial for ASI
- Added section about Stratified PPI into user guide
- Added Stratified PPI scientific validation notebook
- Added `ActiveSampler` in `glide/samplers/` for uncertainty-proportional active sampling
- Added ASI scientific validation notebook
- Added `StratifiedPPIMeanEstimator` for Stratified PPI-based mean estimation
- Added `generate_binary_dataset_with_oracle_sampling` function to simulate ASI-like data
- Added overload for the __getitem__ method for Dataset to support `int`, `slice`, `str`, and `list[str]` keys

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
- Switched documentation layout to full-page width for better readability

### 🐛 Fixed
- Fixed broken badge links in the README

### 💛 Contributors
Thank you to everyone who contributed to this release: @gmartinon-ed, @gherouville-ed, @imerad

---

## [0.1.1] – 2026-03-12

### ✨ Added
- Implemented `PPIMeanEstimator` — the core Prediction-Powered Inference estimator for mean estimation
- Added synthetic dataset generators, including binary data generation, for testing and examples
- Set up MkDocs-based documentation with Read the Docs deployment
- Added a code-quality CI workflow covering linting, type checking, and coverage reporting
- Set up CI/CD pipelines for automated TestPyPI and PyPI publishing
- Added Apache 2.0 licence

### 💛 Contributors
Thank you to everyone who contributed to this release: @gmartinon-ed, @gherouville-ed, @mraki-ed, @vwoelffel-ed, @imerad
