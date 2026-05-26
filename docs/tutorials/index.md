# Tutorials

Each tutorial walks through one complete path in the [decision tree](../user_guide/which_estimator_to_use.md):
from sampling to annotation to estimation, on a simulated dataset.
Use the table below to find the tutorial that matches your situation.

| Cost estimates? | Uncertainty scores? | Stratified data? | Phase 1 sampler | Phase 3 estimator | Tutorial |
|---|---|---|---|---|---|
| No | No | No | Uniform random | PPI++ ¹ | [Standard annotation budget (PPI++)](ppi.ipynb) |
| No | No | Yes | Stratified uniform | Stratified PPI++ ¹ | [Stratified data (Stratified PPI++)](stratified_ppi.ipynb) |
| No | Yes | — | Uncertainty-aware | ASI ¹ | [Uncertainty scores available (ASI)](asi.ipynb) |
| Yes | No | — | Cost-optimal random | PPI++ ¹ | [Cost estimates available (Cost-Optimal Random Sampling)](cost_optimal_random.ipynb) |
| Yes | Yes | — | Cost-optimal | ASI ¹ | [Cost and uncertainty scores available (Cost-Optimal Sampling)](cost_optimal.ipynb) |

¹ Fewer than 50 human labels per stratum: use the PTD variant instead (`PTDMeanEstimator`, `StratifiedPTDMeanEstimator`). The tutorial workflow is identical; only the estimator class changes.
