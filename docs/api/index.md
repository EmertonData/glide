# API Reference

## Estimators

### Classical Methods

| Class | Description |
|-------|-------------|
| [`ClassicalMeanEstimator`](estimators.md#classicalmeanestimator) | Classical sample mean without proxy labels |
| [`StratifiedClassicalMeanEstimator`](estimators.md#stratifiedclassicalmeanestimator) | Classical mean with population-proportional stratification |
| [`IPWClassicalMeanEstimator`](estimators.md#ipwclassicalmeanestimator) | Classical mean with inverse probability weighting |

### Prediction-Powered Methods

| Class | Description |
|-------|-------------|
| [`PPIMeanEstimator`](estimators.md#ppimeanestimator) | Combines labeled data with proxy predictions |
| [`StratifiedPPIMeanEstimator`](estimators.md#stratifiedppimeanestimator) | PPI with per-stratum optimal weighting |
| [`ASIMeanEstimator`](estimators.md#asimeanestimator) | Active statistical inference with non-uniform sampling |
| [`PTDMeanEstimator`](estimators.md#ptdmeanestimator) | Predict-then-debias with bootstrap confidence intervals |

## Confidence Intervals

| Class | Description |
|-------|-------------|
| [`CLTConfidenceInterval`](confidence_intervals.md#cltconfidenceinterval) | CLT-based normal approximation confidence intervals |
| [`BootstrapConfidenceInterval`](confidence_intervals.md#bootstrapconfidenceinterval) | Quantile-based bootstrap confidence intervals |

## Samplers

| Class | Description |
|-------|-------------|
| [`ActiveSampler`](samplers.md#activesampler) | Uncertainty-based active sampling |
| [`StratifiedSampler`](samplers.md#stratifiedsampler) | Stratified budget allocation with Neyman/proportional strategies |

## Data & Results

### Data Utilities

| Module | Description |
|--------|-------------|
| [`glide.simulators`](simulators.md) | Synthetic dataset generators for validation |

### Inference Results

| Class | Description |
|-------|-------------|
| [`ClassicalMeanInferenceResult`](core/mean_inference_results.md#classical-results) | Result object from classical estimators |
| [`PredictionPoweredMeanInferenceResult`](core/mean_inference_results.md#prediction-powered-results) | Result object from prediction-powered estimators |

## I/O

| Module | Description |
|--------|-------------|
| [`glide.io`](io.md) | JSON serialization and export helpers |
