from numpy.typing import NDArray

from glide.estimators.ppi_core import _compute_tuning_parameter


def _compute_clipped_tuning_parameter(
    y_true_labeled: NDArray,
    y_proxy_labeled: NDArray,
    y_proxy_unlabeled: NDArray,
    power_tuning: bool,
    max_tuning_parameter: float,
) -> float:
    tuning_parameter = _compute_tuning_parameter(y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, power_tuning)
    clipped_tuning_parameter = min(max(tuning_parameter, 0.0), max_tuning_parameter)
    return clipped_tuning_parameter
