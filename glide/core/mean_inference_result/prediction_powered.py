from dataclasses import dataclass

from glide.confidence_intervals import ConfidenceInterval
from glide.core.mean_inference_result import MeanInferenceResult


@dataclass(repr=False)
class PredictionPoweredMeanInferenceResult(MeanInferenceResult):
    """Mean inference result for prediction-powered methods."""

    confidence_interval: ConfidenceInterval
    metric_name: str
    estimator_name: str
    n_true: int = 0
    n_proxy: int = 0
    effective_sample_size: int = 0

    def __str__(self) -> str:
        lines = self._common_lines() + [
            f"n_true: {self.n_true}",
            f"n_proxy: {self.n_proxy}",
            f"Effective Sample Size: {self.effective_sample_size:}",
        ]
        return "\n".join(lines)
