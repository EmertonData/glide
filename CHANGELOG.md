# Changelog 📋

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Next release]

### ✨ Added
- `PTDMeanEstimator` for mean estimation with Predict-Then-Debias (bootstrap)
- Section on which estimator to use in the user guide
- `BootStrapConfidenceInterval` and `ConfidenceInterval` Protocol
- `IPWClassicalMeanEstimator` for inverse probability weighted classical mean estimation
- `StratifiedClassicalMeanEstimator` for Stratified classical mean estimation
- Export JSON step in quickstart
- Stratified PPI tutorial notebook
- Code of conduct
- Tutorial for ASI
- Issue templates
- `StratifiedSampler` for optimal per-stratum annotation budget allocation with Neyman and proportional strategies
- Overload for the __setitem__ method for Dataset to support `str`
- Doctest for `glide.io.to_json`
  
### 🔄 Changed
- Improved efficiency of `BootstrapConfidenceInterval.test_null_hypothesis()` from O(n) to O(log n) using binary search on sorted bootstrap samples
- Renamed `compute_lambda` to `compute_tuning_parameter`
- Refactored ASI user to journey to use numpy arrays only
- Refactored stratified PPI to accept numpy arrays as input
- Restructured "User guide" into three pages "Evaluation Workflow", "Samplers" and "Estimators"
- Logo
- Decoupled Stratified PPI from PPI
- `CONTRIBUTING.md` now explains how to setup pre-commits
- Image in the user guide explaining the control sample

### 💛 Contributors
Thank you to everyone who contributed to this release: @gmartinon-ed, @imerad, @mraki-ed


---

## [0.3.0] – 2026-04-03

### ✨ Added
- Section about Stratified PPI into user guide
- Stratified PPI scientific validation notebook
- `StratifiedPPIMeanEstimator` for Stratified PPI-based mean estimation
- Section about ASI into user guide
- ASI scientific validation notebook
- `ActiveSampler` in `glide/samplers/` for uncertainty-proportional active sampling
- `generate_binary_dataset_with_oracle_sampling` function to simulate ASI-like data
- `generate_stratified_binary_dataset` function for generating synthetic stratified binary-label datasets
- Overload for the __getitem__ method for Dataset to support `int`, `slice`, `str`, and `list[str]` keys

### 🔄 Changed
- Raise exception in `PPIMeanEstimator` and `ASIMeanEstimator` when the proxy is constant
- `generate_binary_dataset` now outputs a tuple of labeled/unlabeled dataset
- User guide table of contents and structure
- Getting started section with improved structure
- PPI tutorial improved readability

### 🐛 Fixed
- Useless `EstimatorProtocol` removed

### 💛 Contributors
Thank you to everyone who contributed to this release: @gmartinon-ed, @gherouville-ed, @imerad, @mraki-ed, @awolf-ed, @iben-ed, @msoro-ed

---

## [0.2.0] – 2026-03-19

### ✨ Added
- `ASIMeanEstimator` for Active Statistical Inference-based mean estimation
- `ClassicalMeanEstimator` as a standard baseline estimator
- Power-tuning support (`λ`) to `PPIMeanEstimator` for narrower confidence intervals
- `to_json()` method on inference result objects for easy serialisation
- Quickstart guide to the documentation

### 🔄 Changed
- Refactored `InferenceResult` into dedicated result objects with clearer structure
- Switched documentation layout to full-page width for better readability

### 🐛 Fixed
- Fixed broken badge links in the README

### 💛 Contributors
Thank you to everyone who contributed to this release: @gmartinon-ed, @gherouville-ed, @imerad, @mraki-ed, @vwoelffel-ed

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
