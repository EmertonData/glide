from dataclasses import dataclass

from glide.confidence_intervals import ConfidenceInterval


@dataclass(repr=False)
class MeanInferenceResult:
    """Base class for mean inference results."""

    confidence_interval: ConfidenceInterval
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
            f"Confidence Interval ({confidence_level_pct:.0f}%): [{lower_bound:.3f}, {upper_bound:.3f}]",
            f"Estimator : {self.estimator_name}",
        ]

    def __str__(self) -> str:
        return "\n".join(self._common_lines())

    def summary(self) -> str:
        """Return a formatted summary of the inference result."""
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()
