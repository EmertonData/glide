from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import (
    _validate_bounds,
    _validate_equal_lengths,
    _validate_has_no_nan,
    _validate_unique_clusters,
    _validate_y_true,
)
from glide.estimators.core import _split_labeled_unlabeled


def _preprocess(
    y_true: NDArray,
    y_proxy: NDArray,
    clusters: NDArray,
) -> Tuple[NDArray, NDArray, NDArray]:
    _validate_equal_lengths(y_true, y_proxy, clusters, names=["y_true", "y_proxy", "clusters"])
    _validate_y_true(y_true)
    _validate_has_no_nan(y_proxy, "y_proxy")
    _validate_has_no_nan(clusters, "clusters")

    y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, labeled_mask = _split_labeled_unlabeled(y_true, y_proxy)
    labeled_clusters = clusters[labeled_mask]
    unlabeled_clusters = clusters[~labeled_mask]

    unique_labeled_clusters, labeled_cluster_indices = np.unique(labeled_clusters, return_inverse=True)
    unique_unlabeled_clusters, unlabeled_cluster_indices = np.unique(unlabeled_clusters, return_inverse=True)

    _validate_unique_clusters(unique_labeled_clusters, unique_unlabeled_clusters)

    n_labeled_clusters = len(unique_labeled_clusters)
    n_unlabeled_clusters = len(unique_unlabeled_clusters)

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

    labeled_true_sums = np.bincount(labeled_cluster_indices, weights=y_true_labeled)
    labeled_proxy_sums = np.bincount(labeled_cluster_indices, weights=y_proxy_labeled)
    unlabeled_proxy_sums = np.bincount(unlabeled_cluster_indices, weights=y_proxy_unlabeled)
    labeled_sizes = np.bincount(labeled_cluster_indices)
    unlabeled_sizes = np.bincount(unlabeled_cluster_indices)

    labeled_true_means = labeled_true_sums / labeled_sizes
    labeled_proxy_means = labeled_proxy_sums / labeled_sizes
    unlabeled_proxy_means = unlabeled_proxy_sums / unlabeled_sizes

    return labeled_true_means, labeled_proxy_means, unlabeled_proxy_means
