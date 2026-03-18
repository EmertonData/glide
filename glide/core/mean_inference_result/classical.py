from dataclasses import dataclass

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.mean_inference_result.base import MeanInferenceResult


@dataclass(repr=False)
class ClassicalMeanInferenceResult(MeanInferenceResult):
    confidence_interval: CLTConfidenceInterval
    metric_name: str
    estimator_name: str
    n: int = 0

    def __str__(self) -> str:
        lines = self._common_lines() + [
            f"n: {self.n}",
        ]
        return "\n".join(lines)
