from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import (
    _validate_bounds,
    _validate_equal_lengths,
    _validate_has_no_nan,
    _validate_non_constant,
)


def _preprocess(
    y_true: NDArray,
    y_proxy: NDArray,
    clusters: NDArray,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    _validate_equal_lengths(y_true, y_proxy, clusters, names=["y_true", "y_proxy", "clusters"])
    _validate_has_no_nan(y_proxy, "y_proxy")
    _validate_has_no_nan(clusters, "clusters")

    labeled_mask = ~np.isnan(y_true)
    labeled_clusters = clusters[labeled_mask]
    unlabeled_clusters = clusters[~labeled_mask]

    unique_labeled_clusters, labeled_cluster_indices = np.unique(labeled_clusters, return_inverse=True)
    unique_unlabeled_clusters, unlabeled_cluster_indices = np.unique(unlabeled_clusters, return_inverse=True)

    intersection = np.intersect1d(unique_labeled_clusters, unique_unlabeled_clusters, assume_unique=True)
    _validate_bounds(
        len(intersection),
        "clusters_intersection",
        upper=0,
        error_message=f"Cluster '{intersection[0]}' contains both labeled and unlabeled observations.",
    )

    labeled_true_sums = np.bincount(labeled_cluster_indices, weights=y_true[labeled_mask])
    labeled_proxy_sums = np.bincount(labeled_cluster_indices, weights=y_proxy[labeled_mask])
    unlabeled_proxy_sums = np.bincount(unlabeled_cluster_indices, weights=y_proxy[~labeled_mask])
    labeled_sizes = np.bincount(labeled_cluster_indices)
    unlabeled_sizes = np.bincount(unlabeled_cluster_indices)

    n_labeled_clusters = len(labeled_true_sums)
    n_unlabeled_clusters = len(unlabeled_proxy_sums)

    _validate_bounds(
        n_labeled_clusters,
        "n_labeled_clusters",
        lower=2,
        error_message=f"Need at least 2 fully labeled clusters; got {n_labeled_clusters}.",
    )
    _validate_bounds(
        n_unlabeled_clusters,
        "n_unlabeled_clusters",
        lower=2,
        error_message=f"Need at least 2 fully unlabeled clusters; got {n_unlabeled_clusters}.",
    )

    return (
        labeled_true_sums,
        labeled_proxy_sums,
        unlabeled_proxy_sums,
        labeled_sizes,
        unlabeled_sizes,
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
    _validate_non_constant(
        all_proxy_sums,
        "Proxy cluster sums have zero variance across both labeled and unlabeled clusters; "
        "cannot estimate the tuning parameter.",
    )
    cov = np.cov(labeled_true_sums, labeled_proxy_sums, ddof=1)[0, 1]
    var_proxy = np.var(all_proxy_sums, ddof=1)
    factor = 1 + (n_unlabeled_clusters / n_labeled_clusters) / (labeled_total_size / unlabeled_total_size) ** 2
    denominator = var_proxy * factor
    lambda_ = cov / denominator
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
