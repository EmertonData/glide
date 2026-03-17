from dataclasses import dataclass

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.inference_result.base import MeanInferenceResultBase


@dataclass(repr=False)
class SemiSupervisedMeanInferenceResult(MeanInferenceResultBase):
    confidence_interval: CLTConfidenceInterval
    metric_name: str
    estimator_name: str
    n_true: int = 0
    n_proxy: int = 0
    effective_sample_size: float = 0.0

    def __str__(self) -> str:
        lines = self._common_lines() + [
            f"n_true: {self.n_true}",
            f"n_proxy: {self.n_proxy}",
            f"Effective Sample Size: {self.effective_sample_size:.1f}",
        ]
        return "\n".join(lines)
