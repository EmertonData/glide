from math import floor
from typing import Tuple

from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.validation import (
    _validate_equal_lengths,
    _validate_sample_sizes,
    _validate_y_proxies,
    _validate_y_true,
)
from glide.estimators.classical import ClassicalMeanEstimator
from glide.estimators.core import _split_labeled_unlabeled
from glide.estimators.multi_ppi_core import (
    _compute_mean_estimate,
    _compute_std_estimate,
    _compute_tuning_parameters,
)
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult


class MultiPPIMeanEstimator:
    """Estimator for population mean using Prediction-Powered Inference with multiple proxies.

    This class extends PPIMeanEstimator to settings where M >= 1 proxy predictors are
    available. It finds the optimal tuning parameter vector lambda that minimises the mean
    squared error of the estimate, then applies the PPI correction with that combined
    proxy. This power tuning feature (enabled by default) ensures the estimator is
    always at least as efficient as the naive sample mean, regardless of the quality
    or number of proxies.

    When M = 1, the estimator is equivalent to PPIMeanEstimator with power_tuning=True.

    References
    ----------
    Shan, Jiawei, Zhifeng Chen, Yiming Dong, Yazhen Wang, and Jiwei Zhao.
    "SADA: Safe and Adaptive Aggregation of Multiple Black-Box Predictions in Semi-Supervised Learning."
    arXiv preprint arXiv:2509.21707 (2025).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators import MultiPPIMeanEstimator
    >>> y_true = np.array([5.0, 6.0, np.nan, np.nan])
    >>> y_proxies = np.array([[4.9], [6.1], [5.2], [6.1]])
    >>> estimator = MultiPPIMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxies)
    >>> print(result)
    Metric: Metric
    Point Estimate: 5.618
    Confidence Interval (95%): [4.923, 6.312]
    Estimator : MultiPPIMeanEstimator
    n_true: 2
    n_proxy: 4
    Effective Sample Size: 3
    """

    def _preprocess(
        self,
        y_true_all: NDArray,
        y_proxies_all: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        _validate_equal_lengths(y_true_all, y_proxies_all, names=["y_true", "y_proxies"])
        _validate_y_proxies(y_proxies_all)
        _validate_y_true(y_true_all)
        y_true, y_proxies_labeled, y_proxies_unlabeled, labeled_mask = _split_labeled_unlabeled(
            y_true_all, y_proxies_all
        )
        _validate_sample_sizes(labeled_mask)
        return y_true, y_proxies_labeled, y_proxies_unlabeled

    def estimate(
        self,
        y_true: NDArray,
        y_proxies: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        power_tuning: bool = True,
    ) -> PredictionPoweredMeanInferenceResult:
        """Estimate the population mean using MultiPPI.

        Combines a small set of labeled samples with a large set of unlabeled samples,
        leveraging M proxy predictors simultaneously. The optimal tuning parameter vector lambda
        is estimated from the data and used to form a single combined proxy prediction before
        applying the PPI rectifier.

        Parameters
        ----------
        y_true : NDArray
            Array of observations, shape ``(n_samples,)``.
            Labeled entries are finite; unlabeled entries are ``np.nan``.
        y_proxies : NDArray
            2D array of proxy predictions, shape ``(n_samples, M)``.
            Must be fully populated (no NaN). Each column must have nonzero variance.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        power_tuning : bool, optional
            If ``True`` (default), compute the optimal lambda to minimise the confidence
            interval width. If ``False``, set all tuning parameters to ``1/sqrt(M)``
            to limit proxy variance contribution for large M.

        Returns
        -------
        PredictionPoweredMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"MultiPPIMeanEstimator"``), and the counts
            ``n_true`` (labeled observations) and ``n_proxy`` (all observations).

        Raises
        ------
        ValueError
            - If ``y_true`` and ``y_proxies`` have different lengths.
            - If ``y_proxies`` is not a 2D array.
            - If any value in ``y_proxies`` is NaN.
            - If any column of ``y_proxies`` is constant.
            - If ``y_true`` contains only NaN or its labeled values are constant.
            - If there are fewer than 2 labeled or fewer than 2 unlabeled samples.
            - If the proxy covariance matrix is singular.
        """
        y_true_filtered, y_proxies_labeled, y_proxies_unlabeled = self._preprocess(y_true, y_proxies)
        n_labeled = len(y_true_filtered)
        n_unlabeled = len(y_proxies_unlabeled)
        lambdas_ = _compute_tuning_parameters(y_true_filtered, y_proxies_labeled, y_proxies_unlabeled, power_tuning)
        mean = _compute_mean_estimate(y_true_filtered, y_proxies_labeled, y_proxies_unlabeled, lambdas_)
        std = _compute_std_estimate(y_true_filtered, y_proxies_labeled, y_proxies_unlabeled, lambdas_)
        confidence_interval = CLTConfidenceInterval(
            mean=mean,
            std=std,
            confidence_level=confidence_level,
        )
        classical_confidence_interval = ClassicalMeanEstimator().estimate(y_true_filtered).confidence_interval
        effective_sample_size = floor(n_labeled * classical_confidence_interval.var / confidence_interval.var)
        result = PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_labeled,
            n_proxy=n_labeled + n_unlabeled,
            effective_sample_size=effective_sample_size,
        )
        return result
