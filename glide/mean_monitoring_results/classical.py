from dataclasses import dataclass

from numpy.typing import NDArray

from glide.mean_monitoring_results.base import MeanMonitoringResult


@dataclass(repr=False)
class ClassicalMeanMonitoringResult(MeanMonitoringResult):
    """Monitoring result for the classical (proxy-free) monitor.

    Adds the per-batch labeled sample count to the shared base. ``batch_n[t]``
    is the number of labeled samples in batch ``t`` and has the same length as
    ``batch_mean_estimates``.
    """

    batch_n: NDArray
