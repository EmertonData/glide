from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def _compute_ptd_bootstrap_labeled_means(
    y_true: NDArray,
    y_proxy_labeled: NDArray,
    n_bootstrap: int,
    rng: np.random.Generator,
    pi: Optional[NDArray] = None,
) -> Tuple[NDArray, NDArray]:
    n_labeled = len(y_true)
    if pi is None:
        idx = rng.choice(n_labeled, size=(n_bootstrap, n_labeled), replace=True)
    else:
        weights = pi / pi.sum()
        idx = rng.choice(n_labeled, size=(n_bootstrap, n_labeled), replace=True, p=weights)
    y_true_means = np.mean(y_true[idx], axis=1)
    y_proxy_labeled_means = np.mean(y_proxy_labeled[idx], axis=1)
    return y_true_means, y_proxy_labeled_means


def _compute_ptd_tuning_parameter(
    bootstrap_y_true_means: NDArray,
    bootstrap_y_proxy_labeled_means: NDArray,
    var_proxy_unlabeled: float,
    power_tuning: bool,
) -> float:
    if not power_tuning:
        return 1.0
    cov_matrix = np.cov(bootstrap_y_true_means, bootstrap_y_proxy_labeled_means, ddof=1)
    cov = cov_matrix[0, 1]
    var_proxy_labeled = cov_matrix[1, 1]
    denom = var_proxy_labeled + var_proxy_unlabeled
    lambda_ = cov / denom
    return lambda_


def _compute_ptd_bootstrap_mean_estimates(
    bootstrap_y_true_means: NDArray,
    bootstrap_y_proxy_labeled_means: NDArray,
    mean_proxy_unlabeled: float,
    var_proxy_unlabeled: float,
    lambda_: float,
    rng: np.random.Generator,
) -> NDArray:
    z = rng.standard_normal(len(bootstrap_y_true_means))
    unlabeled_means = mean_proxy_unlabeled + z * np.sqrt(var_proxy_unlabeled)
    rectifier_means = bootstrap_y_true_means - lambda_ * bootstrap_y_proxy_labeled_means
    bootstrap_mean_estimates = lambda_ * unlabeled_means + rectifier_means
    return bootstrap_mean_estimates
