from dataclasses import dataclass
from typing import Protocol

from glide.core.clt_confidence_interval import CLTConfidenceInterval


class MeanInferenceResult(Protocol):
    confidence_interval: CLTConfidenceInterval
    metric_name: str
    estimator_name: str

    @property
    def width(self) -> float:
        return self.confidence_interval.upper_bound - self.confidence_interval.lower_bound

    @property
    def mean(self) -> float:
        return self.confidence_interval.mean

    @property
    def std(self) -> float:
        return self.confidence_interval.std

    def _common_lines(self) -> list:
        lower_bound = self.confidence_interval.lower_bound
        upper_bound = self.confidence_interval.upper_bound
        confidence_level_pct = self.confidence_interval.confidence_level * 100
        return [
            f"Metric: {self.metric_name}",
            f"Point Estimate: {self.mean:.3f}",
            f"Confidence Interval ({confidence_level_pct:.0f}%): [{lower_bound:.2f}, {upper_bound:.2f}]",
            f"Estimator : {self.estimator_name}",
        ]

    def __str__(self) -> str:
        return "\n".join(self._common_lines())

    def summary(self) -> str:
        """Return a formatted summary of the inference result."""
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(repr=False)
class SemiSupervisedMeanInferenceResult(MeanInferenceResult):
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


# Backward-compatibility alias
InferenceResult = SemiSupervisedMeanInferenceResult
