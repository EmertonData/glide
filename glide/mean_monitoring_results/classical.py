from dataclasses import dataclass

from numpy.typing import NDArray

from glide.core.validation import _validate_equal_lengths
from glide.mean_monitoring_results.base import MeanMonitoringResult


@dataclass(repr=False)
class ClassicalMeanMonitoringResult(MeanMonitoringResult):
    """Monitoring result for the classical (proxy-free) monitor.

    Adds the per-batch labeled sample count to the shared base. ``batch_n[t]``
    is the number of labeled samples in batch ``t`` and has the same length as
    ``batch_mean_estimates``.
    """

    batch_n: NDArray

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_equal_lengths(self.batch_mean_estimates, self.batch_n, names=["batch_mean_estimates", "batch_n"])

    def __str__(self) -> str:
        lines = self._common_lines() + [
            f"batch_n: {self.batch_n.sum()}",
        ]
        return "\n".join(lines)
