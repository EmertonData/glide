from math import floor
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.validation import (
    _validate_bounds,
    _validate_equal_lengths,
    _validate_has_no_nan,
    _validate_labeled_unlabeled_clusters,
)
from glide.estimators.cluster_classical import ClusterClassicalMeanEstimator
from glide.estimators.cluster_core import (
    _compute_cluster_mean_estimate,
    _compute_cluster_std_estimate,
    _compute_cluster_tuning_parameter,
)
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult


class ClusterPPIMeanEstimator:
    """Cluster PPI++ estimator for population mean.

    Extends PPI++ mean estimation as in ``PPIMeanEstimator`` to datasets where
    observations are grouped into clusters. Each cluster's true and proxy sums
    are treated as the sampling units, which accounts for within-cluster
    correlation and produces valid confidence intervals under cluster sampling
    designs.

    References
    ----------
    Broska, David. "Cluster-robust PPI reference implementation."
    https://github.com/davidbroska/ppi_py/blob/main/ClusterPPI/mean.py

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators import ClusterPPIMeanEstimator
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan])
    >>> y_proxy = np.array([1.1, 2.2, 3.1, 3.9, 1.5, 1.8, 4.5, 4.8])
    >>> clusters = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
    >>> estimator = ClusterPPIMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy, clusters)
    >>> print(result)
    Metric: Metric
    Point Estimate: 2.744
    Confidence Interval (95%): [1.020, 4.468]
    Estimator : ClusterPPIMeanEstimator
    n_true: 4
    n_proxy: 8
    Effective Sample Size: 5
    """

    def _preprocess(
        self,
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

        _validate_labeled_unlabeled_clusters(unique_labeled_clusters, unique_unlabeled_clusters)

        labeled_true_sums = np.bincount(labeled_cluster_indices, weights=y_true[labeled_mask])
        labeled_proxy_sums = np.bincount(labeled_cluster_indices, weights=y_proxy[labeled_mask])
        unlabeled_proxy_sums = np.bincount(unlabeled_cluster_indices, weights=y_proxy[~labeled_mask])
        labeled_sizes = np.bincount(labeled_cluster_indices)
        unlabeled_sizes = np.bincount(unlabeled_cluster_indices)

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

        return (
            labeled_true_sums,
            labeled_proxy_sums,
            unlabeled_proxy_sums,
            labeled_sizes,
            unlabeled_sizes,
        )

    def estimate(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        clusters: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        power_tuning: bool = True,
    ) -> PredictionPoweredMeanInferenceResult:
        """Estimate the population mean using the Cluster PPI++ estimator.

        Computes cluster sums for labeled and unlabeled clusters and uses them
        as sampling units to apply a PPI++-style bias correction:

            θ̂ = Σ_l u_l / N_L + λ * (Σ_l v_l / N_U - Σ_l s_l / N_L)

            Var(θ̂) = K_L * Var(u_l - λ*s_l, ddof=1) / N_L²
                    + λ² * K_U * Var(v_l, ddof=1) / N_U²

        where ``u_l`` and ``s_l`` are the true and proxy cluster sums for
        labeled clusters, ``v_l`` are the proxy cluster sums for unlabeled
        clusters, ``K_L`` and ``K_U`` are the numbers of labeled and unlabeled
        clusters, and ``N_L``, ``N_U`` are the total labeled and unlabeled
        observation counts.

        Labeled and unlabeled clusters are distinguished by the NaN pattern in
        ``y_true``: a cluster is labeled if every one of its ``y_true`` entries
        is finite, and unlabeled if every entry is ``np.nan``. Partially labeled
        clusters are not supported.

        Parameters
        ----------
        y_true : NDArray
            Array of observations, shape ``(n_samples,)``.
            Labeled entries are finite; unlabeled entries are ``np.nan``.
            All observations in the same cluster must share the same label status.
        y_proxy : NDArray
            Array of proxy predictions, shape ``(n_samples,)``.
            Must be fully populated (no NaN).
        clusters : NDArray
            Array of cluster identifiers, shape ``(n_samples,)``.
            Unique values define the clusters.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        power_tuning : bool, optional
            If ``True`` (default), estimate the optimal power-tuning parameter
            λ from the pooled cluster-level proxy sum variances. If ``False``,
            use λ = 1.0.

        Returns
        -------
        PredictionPoweredMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"ClusterPPIMeanEstimator"``), and the counts
            ``n_true`` (total labeled observations) and ``n_proxy`` (total
            dataset size).

        Raises
        ------
        ValueError
            - If ``y_true``, ``y_proxy``, and ``clusters`` do not all have the
              same length.
            - If any proxy value is NaN.
            - If ``clusters`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If any cluster contains both labeled and unlabeled observations.
            - If fewer than 2 clusters are fully labeled.
            - If fewer than 2 clusters are fully unlabeled.
            - If ``power_tuning=True`` and proxy cluster sums have zero variance across
              both labeled and unlabeled clusters.
        """
        (
            labeled_true_sums,
            labeled_proxy_sums,
            unlabeled_proxy_sums,
            labeled_sizes,
            unlabeled_sizes,
        ) = self._preprocess(y_true, y_proxy, clusters)

        labeled_total_size = np.sum(labeled_sizes)
        unlabeled_total_size = np.sum(unlabeled_sizes)

        lambda_ = _compute_cluster_tuning_parameter(
            labeled_true_sums,
            labeled_proxy_sums,
            unlabeled_proxy_sums,
            labeled_total_size,
            unlabeled_total_size,
            power_tuning,
        )
        mean = _compute_cluster_mean_estimate(
            labeled_true_sums,
            labeled_proxy_sums,
            unlabeled_proxy_sums,
            labeled_total_size,
            unlabeled_total_size,
            lambda_,
        )
        std = _compute_cluster_std_estimate(
            labeled_true_sums,
            labeled_proxy_sums,
            unlabeled_proxy_sums,
            labeled_total_size,
            unlabeled_total_size,
            lambda_,
        )
        confidence_interval = CLTConfidenceInterval(
            mean=mean,
            std=std,
            confidence_level=confidence_level,
        )

        classical_confidence_interval = ClusterClassicalMeanEstimator().estimate(y_true, clusters).confidence_interval
        effective_sample_size = floor(labeled_total_size * classical_confidence_interval.var / confidence_interval.var)
        result = PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=labeled_total_size,
            n_proxy=len(y_proxy),
            effective_sample_size=effective_sample_size,
        )
        return result
