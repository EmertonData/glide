from dataclasses import dataclass

from numpy.typing import NDArray

from glide.core.validation import _validate_equal_lengths
from glide.mean_monitoring_results.base import MeanMonitoringResult


@dataclass(repr=False)
class PredictionPoweredMeanMonitoringResult(MeanMonitoringResult):
    """Monitoring result for the prediction-powered monitor.

    Adds the per-batch labeled and total sample counts to the shared base.
    ``batch_n_true[t]`` is the number of labeled samples in batch ``t`` and
    ``batch_n_proxy[t]`` the total number of samples (labeled plus unlabeled);
    both have the same length as ``batch_mean_estimates``.
    """

    batch_n_true: NDArray
    batch_n_proxy: NDArray

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_equal_lengths(
            self.batch_mean_estimates,
            self.batch_n_true,
            self.batch_n_proxy,
            names=["batch_mean_estimates", "batch_n_true", "batch_n_proxy"],
        )

    def __str__(self) -> str:
        lines = self._common_lines() + [
            f"batch_n_true: {self.batch_n_true[-1]}",
            f"batch_n_proxy: {self.batch_n_proxy[-1]}",
        ]
        return "\n".join(lines)
