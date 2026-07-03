from dataclasses import dataclass

from numpy.typing import NDArray

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
