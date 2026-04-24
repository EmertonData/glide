# API Reference

## Simulators

| Function | Description |
|----------|-------------|
| [`generate_binary_dataset`](simulators.md#glide.simulators.generate_binary_dataset) | Synthetic binary-label dataset |
| [`generate_stratified_binary_dataset`](simulators.md#glide.simulators.generate_stratified_binary_dataset) | Stratified binary-label dataset |
| [`generate_binary_dataset_with_oracle_sampling`](simulators.md#glide.simulators.generate_binary_dataset_with_oracle_sampling) | Binary dataset with oracle sampling probabilities |
| [`generate_gaussian_dataset`](simulators.md#glide.simulators.generate_gaussian_dataset) | Synthetic Gaussian dataset |

## Samplers

| Class | Description |
|-------|-------------|
| [`ActiveSampler`](samplers.md#glide.samplers.active.ActiveSampler) | Uncertainty-based active sampling |
| [`StratifiedSampler`](samplers.md#glide.samplers.stratified.StratifiedSampler) | Stratified budget allocation with Neyman/proportional strategies |
| [`CostOptimalRandomSampler`](samplers.md#glide.samplers.cost_optimal_random.CostOptimalRandomSampler) | Cost-optimal random sampling |

## Estimators

### Classical

| Class | Description |
|-------|-------------|
| [`ClassicalMeanEstimator`](estimators.md#glide.estimators.classical.ClassicalMeanEstimator) | Classical sample mean without proxy labels |
| [`StratifiedClassicalMeanEstimator`](estimators.md#glide.estimators.stratified_classical.StratifiedClassicalMeanEstimator) | Classical mean with population-proportional stratification |
| [`IPWClassicalMeanEstimator`](estimators.md#glide.estimators.ipw_classical.IPWClassicalMeanEstimator) | Classical mean with inverse probability weighting |

### Prediction-Powered

| Class | Description |
|-------|-------------|
| [`PPIMeanEstimator`](estimators.md#glide.estimators.ppi.PPIMeanEstimator) | Combines labeled data with proxy predictions |
| [`StratifiedPPIMeanEstimator`](estimators.md#glide.estimators.stratified_ppi.StratifiedPPIMeanEstimator) | PPI with per-stratum optimal weighting |
| [`ASIMeanEstimator`](estimators.md#glide.estimators.asi.ASIMeanEstimator) | Active statistical inference with non-uniform sampling |
| [`PTDMeanEstimator`](estimators.md#glide.estimators.ptd.PTDMeanEstimator) | Predict-then-debias with bootstrap confidence intervals |
| [`IPWPTDMeanEstimator`](estimators.md#glide.estimators.ipw_ptd.IPWPTDMeanEstimator) | PTD with inverse probability weighting |
| [`StratifiedPTDMeanEstimator`](estimators.md#glide.estimators.stratified_ptd.StratifiedPTDMeanEstimator) | PTD with per-stratum optimal weighting |

## Confidence Intervals

| Class | Description |
|-------|-------------|
| [`CLTConfidenceInterval`](confidence_intervals.md#glide.confidence_intervals.clt.CLTConfidenceInterval) | CLT-based normal approximation confidence intervals |
| [`BootstrapConfidenceInterval`](confidence_intervals.md#glide.confidence_intervals.bootstrap.BootstrapConfidenceInterval) | Quantile-based bootstrap confidence intervals |

## Inference Results

| Class | Description |
|-------|-------------|
| [`ClassicalMeanInferenceResult`](mean_inference_results.md#glide.core.mean_inference_result.classical.ClassicalMeanInferenceResult) | Result object from classical estimators |
| [`PredictionPoweredMeanInferenceResult`](mean_inference_results.md#glide.core.mean_inference_result.prediction_powered.PredictionPoweredMeanInferenceResult) | Result object from prediction-powered estimators |

## I/O

| Module | Description |
|--------|-------------|
| [`glide.io`](io.md) | JSON serialization and export helpers |
