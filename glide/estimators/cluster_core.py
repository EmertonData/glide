import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_non_constant


def _compute_cluster_tuning_parameter(
    labeled_true_sums: NDArray,
    labeled_proxy_sums: NDArray,
    unlabeled_proxy_sums: NDArray,
    labeled_total_size: int,
    unlabeled_total_size: int,
    power_tuning: bool,
) -> float:
    if not power_tuning:
        return 1.0
    n_labeled_clusters = len(labeled_true_sums)
    n_unlabeled_clusters = len(unlabeled_proxy_sums)
    all_proxy_sums = np.hstack([labeled_proxy_sums, unlabeled_proxy_sums])
    _validate_non_constant(
        all_proxy_sums,
        "Proxy cluster sums have zero variance across both labeled and unlabeled clusters; "
        "cannot estimate the tuning parameter.",
    )
    cov = np.cov(labeled_true_sums, labeled_proxy_sums, ddof=1)[0, 1]
    var_proxy = np.var(all_proxy_sums, ddof=1)
    factor = 1 + (n_unlabeled_clusters / n_labeled_clusters) * (labeled_total_size / unlabeled_total_size) ** 2
    lambda_ = cov / (var_proxy * factor)
    return lambda_


def _compute_cluster_mean_estimate(
    labeled_true_sums: NDArray,
    labeled_proxy_sums: NDArray,
    unlabeled_proxy_sums: NDArray,
    labeled_total_size: int,
    unlabeled_total_size: int,
    lambda_: float,
) -> float:
    labeled_true_mean = np.sum(labeled_true_sums) / labeled_total_size
    labeled_proxy_mean = np.sum(labeled_proxy_sums) / labeled_total_size
    unlabeled_proxy_mean = np.sum(unlabeled_proxy_sums) / unlabeled_total_size
    rectifier = labeled_true_mean - lambda_ * labeled_proxy_mean
    mean_estimate = lambda_ * unlabeled_proxy_mean + rectifier
    return mean_estimate


def _compute_cluster_std_estimate(
    labeled_true_sums: NDArray,
    labeled_proxy_sums: NDArray,
    unlabeled_proxy_sums: NDArray,
    labeled_total_size: int,
    unlabeled_total_size: int,
    lambda_: float,
) -> float:
    n_labeled_clusters = len(labeled_true_sums)
    n_unlabeled_clusters = len(unlabeled_proxy_sums)
    rectifier_sums = labeled_true_sums - lambda_ * labeled_proxy_sums
    rectifier_var = n_labeled_clusters * np.var(rectifier_sums, ddof=1) / labeled_total_size**2
    proxy_var = lambda_**2 * n_unlabeled_clusters * np.var(unlabeled_proxy_sums, ddof=1) / unlabeled_total_size**2
    total_var = rectifier_var + proxy_var
    std_estimate = np.sqrt(total_var)
    return std_estimate
