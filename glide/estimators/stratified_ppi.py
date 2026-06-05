from math import floor

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.estimators.ppi_core import (
    _compute_mean_estimate,
    _compute_std_estimate,
    _compute_tuning_parameter,
)
from glide.estimators.stratified_classical import StratifiedClassicalMeanEstimator
from glide.estimators.stratified_core import _preprocess
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult


class StratifiedPPIMeanEstimator:
    """Stratified PPI++ estimator for population mean.

    Extends Prediction-Powered Inference to datasets that are naturally partitioned
    into strata (e.g. by language, domain, or data source). A per-stratum power-tuned
    lambda is computed independently for each stratum, and the final estimate is a
    population-proportional weighted average of the per-stratum PPI++ estimates.

    This yields narrower confidence intervals than standard PPI++ whenever strata differ
    in proxy quality or relative size, because the optimal lambda can adapt to each
    stratum's signal-to-noise ratio.

    References
    ----------
    Fisch, Adam, Joshua Maynez, R. Alex Hofer, Bhuwan Dhingra, Amir Globerson, and
    William W. Cohen. "Stratified prediction-powered inference for effective hybrid
    evaluation of language models." Advances in Neural Information Processing
    Systems 37 (2024): 111489-111514.

    Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona. "A framework
    for efficient model evaluation through stratification, sampling, and estimation."
    In European Conference on Computer Vision, pp. 140-158. Cham: Springer Nature
    Switzerland, 2024.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators import StratifiedPPIMeanEstimator
    >>> y_true = np.array([1.0, 2.0, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan])
    >>> y_proxy = np.array([1.1, 2.2, 1.5, 1.8, 3.9, 5.1, 4.5, 4.8])
    >>> groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> estimator = StratifiedPPIMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy, groups)
    >>> print(result)
    Metric: Metric
    Point Estimate: 3.086
    Confidence Interval (95%): [2.720, 3.452]
    Estimator : StratifiedPPIMeanEstimator
    n_true: 4
    n_proxy: 8
    Effective Sample Size: 14
    """

    def estimate(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        groups: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        power_tuning: bool = True,
    ) -> PredictionPoweredMeanInferenceResult:
        """Estimate the population mean using Stratified PPI++.

        Splits arrays by unique values in ``groups``, computes a power-tuned PPI++
        estimate within each stratum, and combines them with
        population-proportional weights:

            theta = sum_k  w_k * theta_k(lambda_k)
            sigma2 = sum_k  w_k^2 * sigma2_k(lambda_k)

        where ``w_k`` is the fraction of samples in stratum *k*.

        Note that this assumes the portions of labeled vs unlabeled samples are
        approximately the same in all strata which is important for statistical
        validity.

        Labeled and unlabeled samples are distinguished by ``NaN`` in ``y_true``:
        a sample is labeled if its ``y_true`` entry is not ``NaN``.

        Parameters
        ----------
        y_true : NDArray
            Array of observations, shape ``(n_samples,)``.
            Labeled entries are finite; unlabeled entries are ``np.nan``.
        y_proxy : NDArray
            Array of proxy predictions, shape ``(n_samples,)``.
            Must be fully populated (no NaN). Must have nonzero variance.
        groups : NDArray
            Array of integer stratum identifiers, shape ``(n_samples,)``. Unique
            values define the strata.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        power_tuning : bool, optional
            If ``True`` (default), compute the optimal ``lambda_k`` per stratum
            via the PPI++ formula. If ``False``, use ``lambda_k = 1.0`` for all
            strata (classic PPI).

        Returns
        -------
        PredictionPoweredMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"StratifiedPPIMeanEstimator"``), and the counts
            ``n_true`` (total labeled rows) and ``n_proxy`` (total dataset size).

        Raises
        ------
        ValueError
            - If ``groups`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``y_true``, ``y_proxy``, and ``groups`` do not all have the same length.
            - If any proxy value is NaN.
            - If all proxy values within a stratum are identical.
            - If any stratum has fewer than 2 labeled or fewer than 2 unlabeled samples.
        """
        strata = _preprocess(y_true, y_proxy, groups)

        weighted_mean = 0.0
        weighted_var = 0.0
        n_samples = len(y_true)

        for y_true_filtered, y_proxy_labeled, y_proxy_unlabeled in strata:
            stratum_size = len(y_true_filtered) + len(y_proxy_unlabeled)
            w_k = stratum_size / n_samples

            lambda_k = _compute_tuning_parameter(y_true_filtered, y_proxy_labeled, y_proxy_unlabeled, power_tuning)
            mean_k = _compute_mean_estimate(y_true_filtered, y_proxy_labeled, y_proxy_unlabeled, lambda_k)
            std_k = _compute_std_estimate(y_true_filtered, y_proxy_labeled, y_proxy_unlabeled, lambda_k)

            weighted_mean += w_k * mean_k
            weighted_var += w_k**2 * std_k**2

        std = np.sqrt(weighted_var)
        n_true = int(np.sum(~np.isnan(y_true)))

        confidence_interval = CLTConfidenceInterval(
            mean=weighted_mean,
            std=std,
            confidence_level=confidence_level,
        )
        _, stratum_counts = np.unique(groups, return_counts=True)
        stratum_weights = stratum_counts / n_samples
        classical_confidence_interval = (
            StratifiedClassicalMeanEstimator()
            .estimate(y_true, groups, stratum_weights=stratum_weights)
            .confidence_interval
        )
        effective_sample_size = floor(n_true * classical_confidence_interval.var / confidence_interval.var)
        result = PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_true,
            n_proxy=n_samples,
            effective_sample_size=effective_sample_size,
        )
        return result
