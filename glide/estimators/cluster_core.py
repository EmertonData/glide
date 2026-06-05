from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_equal_lengths, _validate_has_no_nan


def _preprocess(
    y_true: NDArray,
    y_proxy: NDArray,
    clusters: NDArray,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    _validate_equal_lengths(y_true, y_proxy, clusters, names=["y_true", "y_proxy", "clusters"])
    _validate_has_no_nan(y_proxy, "y_proxy")
    _validate_has_no_nan(clusters, "clusters")

    labeled_true_means = []
    labeled_proxy_means = []
    unlabeled_proxy_means = []
    labeled_sizes = []
    unlabeled_sizes = []

    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        y_true_cluster = y_true[mask]
        y_proxy_cluster = y_proxy[mask]
        labeled_in_cluster = ~np.isnan(y_true_cluster)

        if labeled_in_cluster.all():
            labeled_true_means.append(np.mean(y_true_cluster))
            labeled_proxy_means.append(np.mean(y_proxy_cluster))
            labeled_sizes.append(len(y_true_cluster))
        elif (~labeled_in_cluster).all():
            unlabeled_proxy_means.append(np.mean(y_proxy_cluster))
            unlabeled_sizes.append(len(y_proxy_cluster))
        else:
            raise ValueError(f"Cluster '{cluster_id}' contains both labeled and unlabeled observations.")

    n_labeled_clusters = len(labeled_true_means)
    n_unlabeled_clusters = len(unlabeled_proxy_means)

    if n_labeled_clusters < 2:
        raise ValueError(f"Need at least 2 fully labeled clusters; got {n_labeled_clusters}.")
    if n_unlabeled_clusters < 2:
        raise ValueError(f"Need at least 2 fully unlabeled clusters; got {n_unlabeled_clusters}.")

    return (
        np.array(labeled_true_means),
        np.array(labeled_proxy_means),
        np.array(unlabeled_proxy_means),
        np.array(labeled_sizes),
        np.array(unlabeled_sizes),
    )


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
    cov_tu = np.cov(labeled_true_sums, labeled_proxy_sums, ddof=1)[0, 1]
    var_proxy = np.var(all_proxy_sums, ddof=1)
    numerator = n_labeled_clusters * cov_tu / labeled_total_size**2
    weight_sum = n_labeled_clusters / labeled_total_size**2 + n_unlabeled_clusters / unlabeled_total_size**2
    denominator = var_proxy * weight_sum
    if denominator == 0.0:
        raise ValueError(
            "Proxy cluster sums have zero variance across both labeled and unlabeled clusters; "
            "cannot estimate the tuning parameter."
        )
    lambda_ = numerator / denominator
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
    labeled_var = n_labeled_clusters * np.var(rectifier_sums, ddof=1) / labeled_total_size**2
    unlabeled_var = lambda_**2 * n_unlabeled_clusters * np.var(unlabeled_proxy_sums, ddof=1) / unlabeled_total_size**2
    total_var = labeled_var + unlabeled_var
    std_estimate = np.sqrt(total_var)
    return std_estimate
