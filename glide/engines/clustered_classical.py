from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_bounds, _validate_equal_lengths, _validate_has_no_nan
from glide.engines.classical import ClassicalMeanEngine
from glide.engines.clustered_core import _compute_cluster_means


class ClusteredClassicalMeanEngine:
    def __init__(self) -> None:
        self._classical_engine = ClassicalMeanEngine()

    def preprocess(self, y: NDArray, clusters: NDArray) -> NDArray:
        _validate_equal_lengths(y, clusters, names=["y", "clusters"])
        _validate_has_no_nan(clusters, "clusters")
        not_nan_mask = ~np.isnan(y)
        y_valid = y[not_nan_mask]
        unique_valid_clusters, cluster_indices = np.unique(clusters[not_nan_mask], return_inverse=True)
        n_valid_clusters = len(unique_valid_clusters)
        _validate_bounds(
            n_valid_clusters,
            "n_valid_clusters",
            lower=2,
            error_message=f"Need at least 2 clusters with non-NaN observations; got {n_valid_clusters}.",
        )
        cluster_means = _compute_cluster_means(y_valid, cluster_indices)
        return cluster_means

    def fit_tuning_parameter(self, dataset: NDArray, *, power_tuning: bool) -> None:
        return None

    def compute_mean_and_std(self, dataset: NDArray, tuning_parameter: None) -> Tuple[float, float]:
        mean, std = self._classical_engine.compute_mean_and_std(dataset, tuning_parameter)
        return mean, std
