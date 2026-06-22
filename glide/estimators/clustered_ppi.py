from math import floor

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.estimators.clustered_classical import ClusteredClassicalMeanEstimator
from glide.estimators.clustered_core import _preprocess
from glide.estimators.ppi_core import _compute_mean_estimate, _compute_std_estimate, _compute_tuning_parameter
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult


class ClusteredPPIMeanEstimator:
    """Clustered PPI++ estimator for population mean.

    Extends PPI++ mean estimation as in ``PPIMeanEstimator`` to datasets where
    observations are grouped into clusters. Each cluster's true and proxy means
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
    >>> from glide.estimators import ClusteredPPIMeanEstimator
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan])
    >>> y_proxy = np.array([1.1, 2.2, 3.1, 3.9, 1.5, 1.8, 4.5, 4.8])
    >>> clusters = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
    >>> estimator = ClusteredPPIMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy, clusters)
    >>> print(result)
    Metric: Metric
    Point Estimate: 2.744
    Confidence Interval (95%): [1.020, 4.468]
    Estimator : ClusteredPPIMeanEstimator
    n_true: 4
    n_proxy: 8
    Effective Sample Size: 5
    """

    def estimate(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        clusters: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        power_tuning: bool = True,
    ) -> PredictionPoweredMeanInferenceResult:
        """Estimate the population mean using the Clustered PPI++ estimator.

        Computes cluster means for labeled and unlabeled clusters and uses them
        as sampling units to apply a PPI++-style bias correction:

            θ̂ = mean(u_l) + λ * (mean(v_l) - mean(s_l))

            Var(θ̂) = Var(u_l - λ*s_l, ddof=1) / M_L
                    + λ² * Var(v_l, ddof=1) / M_U

        where ``u_l`` and ``s_l`` are the true and proxy cluster means for
        labeled clusters, ``v_l`` are the proxy cluster means for unlabeled
        clusters, and ``M_L``, ``M_U`` are the numbers of labeled and unlabeled
        clusters.

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
            λ from the pooled cluster-level proxy mean variances. If ``False``,
            use λ = 1.0.

        Returns
        -------
        PredictionPoweredMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"ClusteredPPIMeanEstimator"``), and the counts
            ``n_true`` (total labeled observations) and ``n_proxy`` (total
            dataset size).

        Raises
        ------
        ValueError
            - If ``y_true``, ``y_proxy``, and ``clusters`` do not all have the
              same length.
            - If labeled ``y_true`` values are constant.
            - If any proxy value is NaN.
            - If ``clusters`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If any cluster contains both labeled and unlabeled observations.
            - If fewer than 2 clusters are fully labeled.
            - If fewer than 2 clusters are fully unlabeled.
            - If ``power_tuning=True`` and proxy cluster means have zero variance across
              both labeled and unlabeled clusters.
        """
        (
            labeled_true_means,
            labeled_proxy_means,
            unlabeled_proxy_means,
        ) = _preprocess(y_true, y_proxy, clusters)

        _lambda = _compute_tuning_parameter(
            labeled_true_means, labeled_proxy_means, unlabeled_proxy_means, power_tuning
        )
        mean = _compute_mean_estimate(labeled_true_means, labeled_proxy_means, unlabeled_proxy_means, _lambda)
        std = _compute_std_estimate(labeled_true_means, labeled_proxy_means, unlabeled_proxy_means, _lambda)
        confidence_interval = CLTConfidenceInterval(
            mean=mean,
            std=std,
            confidence_level=confidence_level,
        )

        labeled_total_size = np.sum(~np.isnan(y_true))
        classical_confidence_interval = ClusteredClassicalMeanEstimator().estimate(y_true, clusters).confidence_interval
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
