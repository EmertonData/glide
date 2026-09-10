from math import floor

from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.validation import _validate_non_constant
from glide.engines.classical import ClassicalMeanEngine
from glide.engines.multi_ppi import MultiPPIMeanEngine
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

    def __init__(self) -> None:
        self._engine = MultiPPIMeanEngine()
        self._classical_engine = ClassicalMeanEngine()

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
            Must be fully populated (no NaN). Each column must have nonzero variance when
            ``power_tuning=True``.
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
            - If any column of ``y_proxies`` is constant (with ``power_tuning=True``).
            - If ``y_true`` labeled values are constant.
            - If there are fewer than 2 labeled or fewer than 2 unlabeled samples.
            - If the proxy covariance matrix is singular.
        """
        multi_ppi_dataset = self._engine.preprocess(y_true, y_proxies)
        y_true_labeled, _, y_proxies_unlabeled = multi_ppi_dataset
        _validate_non_constant(y_true_labeled, "'y_true' labeled values are constant.")

        tuning_parameter = self._engine.fit_tuning_parameter(multi_ppi_dataset, power_tuning)
        mean, std = self._engine.compute_mean_and_std(multi_ppi_dataset, tuning_parameter)
        confidence_interval = CLTConfidenceInterval(
            mean=mean,
            std=std,
            confidence_level=confidence_level,
        )

        _, classical_std = self._classical_engine.compute_mean_and_std(y_true_labeled, None)
        n_labeled, n_unlabeled = len(y_true_labeled), len(y_proxies_unlabeled)
        effective_sample_size = floor(n_labeled * classical_std**2 / std**2)

        result = PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_labeled,
            n_proxy=n_labeled + n_unlabeled,
            effective_sample_size=effective_sample_size,
        )
        return result
