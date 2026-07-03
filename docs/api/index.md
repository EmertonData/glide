# API Reference

## Simulators

| Function | Description |
|----------|-------------|
| [`generate_binary_dataset`](simulators.md#glide.simulators.generate_binary_dataset) | Synthetic binary-label dataset |
| [`generate_stratified_binary_dataset`](simulators.md#glide.simulators.generate_stratified_binary_dataset) | Stratified binary-label dataset |
| [`generate_binary_dataset_with_oracle_sampling`](simulators.md#glide.simulators.generate_binary_dataset_with_oracle_sampling) | Binary dataset with oracle sampling probabilities |
| [`generate_gaussian_dataset`](simulators.md#glide.simulators.generate_gaussian_dataset) | Synthetic Gaussian dataset |
| [`generate_clustered_binary_dataset`](simulators.md#glide.simulators.generate_clustered_binary_dataset) | Synthetic clustered binary-label dataset |
| [`generate_multi_binary_dataset`](simulators.md#glide.simulators.generate_multi_binary_dataset) | Synthetic binary-label dataset with multiple proxies |
| [`simulate_annotation`](simulators.md#glide.simulators.simulate_annotation) | Simulate annotation in the simulation lifecycle |

## Samplers

| Class | Description |
|-------|-------------|
| [`UniformSampler`](samplers.md#glide.samplers.uniform.UniformSampler) | Uniform random sampling |
| [`ActiveSampler`](samplers.md#glide.samplers.active.ActiveSampler) | Uncertainty-based active sampling |
| [`StratifiedSampler`](samplers.md#glide.samplers.stratified.StratifiedSampler) | Stratified budget allocation with Neyman/proportional strategies |
| [`CostOptimalRandomSampler`](samplers.md#glide.samplers.cost_optimal_random.CostOptimalRandomSampler) | Cost-optimal random sampling |
| [`CostOptimalSampler`](samplers.md#glide.samplers.cost_optimal.CostOptimalSampler) | Uncertainty-based cost-optimal sampling |
| [`UniformClusteredSampler`](samplers.md#glide.samplers.clustered.UniformClusteredSampler) | Uniform clustered random sampling |

## Estimators

### Classical

| Class | Description |
|-------|-------------|
| [`ClassicalMeanEstimator`](estimators.md#glide.estimators.classical.ClassicalMeanEstimator) | Classical sample mean without proxy labels |
| [`StratifiedClassicalMeanEstimator`](estimators.md#glide.estimators.stratified_classical.StratifiedClassicalMeanEstimator) | Classical mean with population-proportional stratification |
| [`IPWClassicalMeanEstimator`](estimators.md#glide.estimators.ipw_classical.IPWClassicalMeanEstimator) | Classical mean with inverse probability weighting |
| [`ClusteredClassicalMeanEstimator`](estimators.md#glide.estimators.clustered_classical.ClusteredClassicalMeanEstimator) | Classical sample mean on clustered data |

### Prediction-Powered

| Class | Description |
|-------|-------------|
| [`PPIMeanEstimator`](estimators.md#glide.estimators.ppi.PPIMeanEstimator) | Combines labeled data with proxy predictions |
| [`StratifiedPPIMeanEstimator`](estimators.md#glide.estimators.stratified_ppi.StratifiedPPIMeanEstimator) | PPI with per-stratum optimal weighting |
| [`ClusteredPPIMeanEstimator`](estimators.md#glide.estimators.clustered_ppi.ClusteredPPIMeanEstimator) | PPI for clustered data |
| [`ASIMeanEstimator`](estimators.md#glide.estimators.asi.ASIMeanEstimator) | Active statistical inference with non-uniform sampling |
| [`MultiPPIMeanEstimator`](estimators.md#glide.estimators.multi_ppi.MultiPPIMeanEstimator) | Combines labeled data with predictions from multiple proxies |
| [`PTDMeanEstimator`](estimators.md#glide.estimators.ptd.PTDMeanEstimator) | Predict-then-debias with bootstrap confidence intervals |
| [`StratifiedPTDMeanEstimator`](estimators.md#glide.estimators.stratified_ptd.StratifiedPTDMeanEstimator) | PTD with per-stratum optimal weighting |
| [`ClusteredPTDMeanEstimator`](estimators.md#glide.estimators.clustered_ptd.ClusteredPTDMeanEstimator) | PTD for clustered data |
| [`IPWPTDMeanEstimator`](estimators.md#glide.estimators.ipw_ptd.IPWPTDMeanEstimator) | PTD with inverse probability weighting |

## Confidence Intervals

| Class | Description |
|-------|-------------|
| [`CLTConfidenceInterval`](confidence_intervals.md#glide.confidence_intervals.clt.CLTConfidenceInterval) | CLT-based normal approximation confidence intervals |
| [`BootstrapConfidenceInterval`](confidence_intervals.md#glide.confidence_intervals.bootstrap.BootstrapConfidenceInterval) | Quantile-based bootstrap confidence intervals |

## Confidence Sequences

| Class | Description |
|-------|-------------|
| [`EmpiricalBernsteinConfidenceSequence`](confidence_sequences.md#glide.confidence_sequences.empirical_bernstein.EmpiricalBernsteinConfidenceSequence) | Anytime-valid empirical-Bernstein confidence sequence |

## Inference Results

| Class | Description |
|-------|-------------|
| [`ClassicalMeanInferenceResult`](mean_inference_results.md#glide.mean_inference_results.classical.ClassicalMeanInferenceResult) | Result object from classical estimators |
| [`PredictionPoweredMeanInferenceResult`](mean_inference_results.md#glide.mean_inference_results.prediction_powered.PredictionPoweredMeanInferenceResult) | Result object from prediction-powered estimators |

## Scientific Validation

| Function | Description |
|----------|-------------|
| [`run_monte_carlo`](scientific_validation.md#glide.scientific_validation.run_monte_carlo) | Monte Carlo driver for coverage and efficiency validation |
| [`compute_hits`](scientific_validation.md#glide.scientific_validation.compute_hits) | Per-seed hit indicators for coverage computation |
| [`coverage_with_error_bar`](scientific_validation.md#glide.scientific_validation.coverage_with_error_bar) | Empirical coverage and confidence interval |

## I/O

| Module | Description |
|--------|-------------|
| [`glide.io`](io.md) | JSON serialization and export helpers |
