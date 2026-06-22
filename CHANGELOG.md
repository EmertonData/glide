# Changelog 📋

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Next release]

### ✨ Added
- `ClusteredPTDMeanEstimator`: cluster-robust Predict-Then-Debias estimator for clsutered data.
- `generate_multi_binary_dataset` simulator for generating binary oracle datasets with multiple proxies.
- Spider Text-to-SQL case study

### 🔄 Changed
- Renamed `ClusterPPIMeanEstimator` → `ClusteredPPIMeanEstimator`, `ClusterClassicalMeanEstimator` → `ClusteredClassicalMeanEstimator`, `UniformClusterSampler` → `UniformClusteredSampler`.

### 🐛 Fixed

### 💛 Contributors

## [0.7.0] – 2026-06-12

### ✨ Added
- Cluster-level inference support: `ClusterPPIMeanEstimator`, `ClusterClassicalMeanEstimator`, `UniformClusterSampler`, and `generate_clustered_binary_dataset` for end-to-end prediction-powered inference on clustered data.
- Diagrams in the README and user guide illustrating prediction-powered inference and GLIDE's three-step workflow.

### 🔄 Changed
- `generate_gaussian_dataset` now accepts `n_total` instead of `n_labeled` and `n_unlabeled`.
- Renamed sampler parameters for clarity: `budget` → `n_samples` in `UniformSampler`, `StratifiedSampler`, and `ActiveSampler`; `budget` → `max_cost` in `CostOptimalSampler` and `CostOptimalRandomSampler`.

### 🐛 Fixed
- Fixed broken quickstart link in README.

### 💛 Contributors
Thank you to everyone who contributed to this release: @gmartinon-ed, @imerad

## [0.6.0] – 2026-06-01

### ✨ Added
- `scientific_validation.md` deep-dive page covering shared validation methodology
- `CostOptimalSampler` for uncertainty-based cost-optimal annotation strategy

### 🔄 Changed
- Centralised input validation logic into `glide.core.validation`
- Deep-dive validation notebooks use shared utilities from `glide.scientific_validation`
- Binary dataset generators now take `n_total` instead of `n_labeled` + `n_unlabeled`, and return fully populated oracle arrays.
- Improved `ActiveSampler` to compute probabilities that always sum to the budget.

### 🐛 Fixed
- Fixed effective sample size computation to use the correct classical baseline for each sampling design
- Invalid inputs/outputs in `StratifiedSampler` with no raised errors

### 💛 Contributors
Thank you to everyone who contributed to this release: @gmartinon-ed, @imerad, @mraki-ed

## [0.5.0] – 2026-05-18

### ✨ Added
- `simulate_annotation` for simulating annotation in the simulation lifecycle
- `UniformSampler` for uniform random sampling
- `IPWPTDMeanEstimator` for inverse probability weighted PTD mean estimation
- `CostOptimalRandomSampler` for cost-optimal annotation strategy
- `StratifiedPTDMeanEstimator` stratified extension of `PTDMeanEstimator`

### 🔄 Changed
- Moved `glide.core.mean_inference_result` to `glide.mean_inference_results` and removed `glide.core`
- "Which estimator to choose" decision tree updated with samplers
- `StratifiedSampler` now returns sampling indicators only
- Refactored `glide.core.simulated_datasets` into dedicated `glide.simulators` module with separate files for each generator function
- Replaced "semi-supervised" with "prediction-powered" everywhere in the docs and code

### 🐛 Fixed
- Handling of NaN values in `StratifiedClassicalMeanEstimator`

### 💛 Contributors
Thank you to everyone who contributed to this release: @gmartinon-ed, @imerad, @mraki-ed

---

## [0.4.0] – 2026-04-20

### ✨ Added
- Examples section in the documentation
- Section on which estimator to use in the user guide
- `PTDMeanEstimator` for mean estimation with Predict-Then-Debias (bootstrap)
- `BootStrapConfidenceInterval` and `ConfidenceInterval` Protocol
- `IPWClassicalMeanEstimator` for inverse probability weighted classical mean estimation
- `StratifiedClassicalMeanEstimator` for Stratified classical mean estimation
- `StratifiedSampler` for optimal per-stratum annotation budget allocation with Neyman and proportional strategies
- Tutorial for Stratified PPI
- Tutorial for ASI
- Code of conduct
- Issue templates
- Doctest for `glide.io.to_json`

### 🔄 Changed
- Removed dataset container object `Dataset`
- Improved efficiency of `BootstrapConfidenceInterval.test_null_hypothesis()` from O(n) to O(log n) using binary search on sorted bootstrap samples
- Refactored ASI to use numpy arrays only
- Refactored stratified PPI to accept numpy arrays as input
- Restructured "User guide" into three pages "Evaluation Workflow", "Samplers" and "Estimators"
- Logo
- `CONTRIBUTING.md` now explains how to setup pre-commits
- Image in the user guide explaining the control sample

### 🐛 Fixed
- CI for notebooks execution

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