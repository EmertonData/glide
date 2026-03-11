from dataclasses import dataclass

from glide.core.clt_confidence_interval import CLTConfidenceInterval


@dataclass
class InferenceResult:
    result: CLTConfidenceInterval
    metric_name: str
    estimator_name: str
    n_true: int
    n_proxy: int

    def __str__(self) -> str:
        lower_bound = self.result.lower_bound
        upper_bound = self.result.upper_bound
        confidence_level_pct = self.result.confidence_level * 100
        return "\n".join(
            [
                f"Metric: {self.metric_name}",
                f"Point Estimate: {self.result.mean:.3f}",
                f"Confidence Interval ({confidence_level_pct:.0f}%): [{lower_bound:.2f}, {upper_bound:.2f}]",
                f"Estimator : {self.estimator_name}",
                f"n_true: {self.n_true}",
                f"n_proxy: {self.n_proxy}",
            ]
        )

    def summary(self) -> str:
        """Return a formatted summary of the inference result."""
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()
