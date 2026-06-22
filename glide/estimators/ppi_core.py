import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_non_constant


def _compute_tuning_parameter(
    y_true: NDArray,
    y_proxy_labeled: NDArray,
    y_proxy_unlabeled: NDArray,
    power_tuning: bool,
) -> float:
    if not power_tuning:
        return 1.0
    n_labeled = len(y_true)
    n_unlabeled = len(y_proxy_unlabeled)
    y_proxy_all = np.hstack([y_proxy_labeled, y_proxy_unlabeled])
    _validate_non_constant(
        y_proxy_all,
        "Proxy labels have zero variance; cannot estimate the tuning parameter.",
    )
    cov = np.cov(y_true, y_proxy_labeled, ddof=1)[0, 1]
    var = np.var(y_proxy_all, ddof=1)
    factor = 1 + n_labeled / n_unlabeled
    lambda_ = cov / (factor * var)
    return lambda_


def _compute_mean_estimate(
    y_true: NDArray,
    y_proxy_labeled: NDArray,
    y_proxy_unlabeled: NDArray,
    lambda_: float,
) -> float:
    rectifier = np.mean(y_true) - lambda_ * np.mean(y_proxy_labeled)
    proxy_mean = lambda_ * np.mean(y_proxy_unlabeled)
    mean_estimate = proxy_mean + rectifier
    return mean_estimate


def _compute_std_estimate(
    y_true: NDArray,
    y_proxy_labeled: NDArray,
    y_proxy_unlabeled: NDArray,
    lambda_: float,
) -> float:
    n_labeled, n_unlabeled = len(y_true), len(y_proxy_unlabeled)
    rectifier_residuals = y_true - lambda_ * y_proxy_labeled
    rectifier_var = np.var(rectifier_residuals, ddof=1) / n_labeled
    proxy_projections = lambda_ * y_proxy_unlabeled
    proxy_var = np.var(proxy_projections, ddof=1) / n_unlabeled
    var_estimate = rectifier_var + proxy_var
    std_estimate = np.sqrt(var_estimate)
    return std_estimate
