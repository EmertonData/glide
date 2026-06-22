from math import floor
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import BootstrapConfidenceInterval
from glide.estimators.clustered_classical import ClusteredClassicalMeanEstimator
from glide.estimators.clustered_core import _preprocess
from glide.estimators.ptd_core import (
    _compute_bootstrap_labeled_means,
    _compute_bootstrap_mean_estimates,
    _compute_tuning_parameter,
)
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult


class ClusteredPTDMeanEstimator:
    """Clustered Predict-Then-Debias estimator for population mean.

    Extends PTD to datasets where observations are grouped into clusters and
    each cluster is either entirely labeled or entirely unlabeled. The bootstrap
    resamples whole clusters rather than individual observations, which accounts
    for within-cluster correlation and produces valid confidence intervals under
    cluster sampling designs.

    A power-tuning parameter λ is estimated from the joint bootstrap covariance
    of the labeled cluster means and labeled cluster proxy means. The final
    confidence interval is a percentile interval over B bootstrap replicates of:

        θ_b = mean(y_true_b) + λ * (mean(y_proxy_unlabeled_b) - mean(y_proxy_labeled_b))

    where each _b subscript denotes a size-weighted resample of cluster-level means.

    References
    ----------
    Kluger, Dan M., Kerri Lu, Tijana Zrnic, Sherrie Wang, and Stephen Bates.
    "Prediction-powered inference with imputed covariates and nonuniform sampling."
    arXiv preprint arXiv:2501.18577 (2025).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators import ClusteredPTDMeanEstimator
    >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan, np.nan, np.nan])
    >>> y_proxy = np.array([1.1, 2.2, 3.1, 3.9, 1.5, 1.8, 4.5, 4.8])
    >>> clusters = np.array(["A", "A", "B", "B", "C", "C", "D", "D"])
    >>> estimator = ClusteredPTDMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy, clusters, n_bootstrap=5, random_seed=0)
    >>> print(result)
    Metric: Metric
    Point Estimate: 2.517
    Confidence Interval (95%): [1.657, 3.497]
    Estimator : ClusteredPTDMeanEstimator
    n_true: 4
    n_proxy: 8
    Effective Sample Size: 6
    """

    def estimate(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        clusters: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        n_bootstrap: int = 2000,
        power_tuning: bool = True,
        random_seed: Optional[int] = None,
    ) -> PredictionPoweredMeanInferenceResult:
        """Estimate the population mean using the Clustered Predict-Then-Debias bootstrap.

        Aggregates observations into cluster-level means, bootstraps labeled cluster
        means with size-weighting, and assembles per-replicate PTD estimates to form
        a percentile confidence interval. The unlabeled proxy mean's sampling
        variability is approximated by a Gaussian draw per replicate, computed from
        cluster sums so that larger clusters contribute proportionally more.

        Labeled and unlabeled clusters are distinguished by the NaN pattern in
        ``y_true``: a cluster is labeled if every one of its ``y_true`` entries is
        finite, and unlabeled if every entry is ``np.nan``. Partially labeled clusters
        are not supported.

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
        n_bootstrap : int, optional
            Number of bootstrap resamples. Defaults to ``2000``.
        power_tuning : bool, optional
            If ``True`` (default), estimate the optimal power-tuning parameter
            λ from the bootstrap covariances. If ``False``, use λ = 1.0.
        random_seed : int, optional
            Seed for the random number generator. Defaults to ``None``.

        Returns
        -------
        PredictionPoweredMeanInferenceResult
            Contains the bootstrap-based confidence interval, the metric name,
            the estimator name (``"ClusteredPTDMeanEstimator"``), and the counts
            ``n_true`` (total labeled observations) and ``n_proxy`` (total dataset size).

        Raises
        ------
        ValueError
            - If ``y_true``, ``y_proxy``, and ``clusters`` do not all have the same length.
            - If any proxy value is NaN.
            - If ``clusters`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If any cluster contains both labeled and unlabeled observations.
            - If fewer than 2 clusters are fully labeled.
            - If fewer than 2 clusters are fully unlabeled.
        """
        (
            labeled_true_means,
            labeled_proxy_means,
            unlabeled_proxy_means,
        ) = _preprocess(y_true, y_proxy, clusters)

        n_unlabeled_clusters = len(unlabeled_proxy_means)
        mean_proxy_unlabeled = np.mean(unlabeled_proxy_means)
        var_proxy_unlabeled = np.var(unlabeled_proxy_means, ddof=1) / n_unlabeled_clusters

        rng = np.random.default_rng(random_seed)

        bootstrap_y_true_means, bootstrap_y_proxy_labeled_means = _compute_bootstrap_labeled_means(
            labeled_true_means, labeled_proxy_means, n_bootstrap, rng
        )
        lambda_ = _compute_tuning_parameter(
            bootstrap_y_true_means, bootstrap_y_proxy_labeled_means, var_proxy_unlabeled, power_tuning
        )
        bootstrap_estimates = _compute_bootstrap_mean_estimates(
            bootstrap_y_true_means,
            bootstrap_y_proxy_labeled_means,
            mean_proxy_unlabeled,
            var_proxy_unlabeled,
            lambda_,
            rng,
        )

        confidence_interval = BootstrapConfidenceInterval(
            bootstrap_estimates=bootstrap_estimates,
            confidence_level=confidence_level,
        )
        n_labeled = np.sum(~np.isnan(y_true))
        classical_confidence_interval = ClusteredClassicalMeanEstimator().estimate(y_true, clusters).confidence_interval
        effective_sample_size = floor(n_labeled * classical_confidence_interval.var / confidence_interval.var)
        result = PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_labeled,
            n_proxy=len(y_proxy),
            effective_sample_size=effective_sample_size,
        )
        return result
