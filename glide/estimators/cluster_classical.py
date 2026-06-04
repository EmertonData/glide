from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.validation import _validate_bounds, _validate_equal_lengths
from glide.mean_inference_results import ClassicalMeanInferenceResult


class ClusterClassicalMeanEstimator:
    """Cluster classical estimator for population mean.

    Extends mean estimation as in ``ClassicalMeanEstimator`` to datasets where
    observations are grouped into clusters. Each cluster's size-weighted
    contribution is treated as the sampling unit, which accounts for
    within-cluster correlation and produces valid confidence intervals under
    cluster sampling designs.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators import ClusterClassicalMeanEstimator
    >>> y = np.array([5.0, 5.0, 7.0, 7.0])
    >>> clusters = np.array(["A", "A", "B", "B"])
    >>> estimator = ClusterClassicalMeanEstimator()
    >>> result = estimator.estimate(y, clusters)
    >>> print(result)
    Metric: Metric
    Point Estimate: 6.000
    Confidence Interval (95%): [4.040, 7.960]
    Estimator : ClusterClassicalMeanEstimator
    n: 4
    """

    def _preprocess(
        self,
        y: NDArray,
        clusters: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        _validate_equal_lengths(y, clusters, names=["y", "clusters"])
        not_nan_mask = ~np.isnan(y)
        y_valid = y[not_nan_mask]
        clusters_valid = clusters[not_nan_mask]

        unique_valid_clusters = np.unique(clusters_valid)
        n_valid_clusters = len(unique_valid_clusters)
        _validate_bounds(
            n_valid_clusters,
            "n_valid_clusters",
            lower=2,
            error_message=f"Need at least 2 clusters with non-NaN observations; got {n_valid_clusters}.",
        )
        return y_valid, clusters_valid, unique_valid_clusters

    def estimate(
        self,
        y: NDArray,
        clusters: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
    ) -> ClassicalMeanInferenceResult:
        """Estimate the population mean using the cluster classical estimator.

        Aggregates observations into cluster-level means weighted by cluster
        size, then applies the CLT to the cluster sums as sampling units:

            theta = sum_l  (n_l * mu_l) / N
            sigma2 = L * Var(u_l, ddof=1) / N^2

        where ``u_l = n_l * mu_l`` are the cluster sums, ``L`` is the number
        of clusters, and ``N = sum_l n_l`` is the total number of observations.
        NaN values in ``y`` are dropped before making the computations. Clusters
        that contain only NaN are not used.

        Parameters
        ----------
        y : NDArray
            Array of observations, shape ``(n_samples,)``. NaN values are
            treated as missing and dropped per cluster.
        clusters : NDArray
            Array of cluster identifiers, shape ``(n_samples,)``.
            Unique values define the clusters.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.

        Returns
        -------
        ClassicalMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"ClusterClassicalMeanEstimator"``), and ``n``
            (total number of non-NaN observations across all clusters).

        Raises
        ------
        ValueError
            - If ``y`` and ``clusters`` do not have the same length.
            - If fewer than 2 clusters have at least one non-NaN observation.
        """
        y_valid, clusters_valid, unique_valid_clusters = self._preprocess(y, clusters)
        n_valid_clusters = len(unique_valid_clusters)

        sums = np.zeros(n_valid_clusters)

        for i, cluster_id in enumerate(unique_valid_clusters):
            cluster_mask = clusters_valid == cluster_id
            cluster_y_valid = y_valid[cluster_mask]
            sums[i] = np.sum(cluster_y_valid)

        total_size = len(y_valid)

        mean = np.sum(sums) / total_size
        var = n_valid_clusters * np.var(sums, ddof=1) / total_size**2
        std = np.sqrt(var)

        ci = CLTConfidenceInterval(
            mean=mean,
            std=std,
            confidence_level=confidence_level,
        )
        result = ClassicalMeanInferenceResult(
            confidence_interval=ci,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n=total_size,
        )
        return result
