import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.mean_inference_result import ClassicalMeanInferenceResult


class ClassicalMeanEstimator:
    """Estimator for population mean using the classical sample mean.

    Uses only a single array ``y`` to compute the sample mean and its
    standard error via the Central Limit Theorem. This serves as a baseline
    that does not require proxy predictions.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators.classical import ClassicalMeanEstimator
    >>> y = np.array([5.0, 6.0, 4.0, 7.0])
    >>> estimator = ClassicalMeanEstimator()
    >>> result = estimator.estimate(y)
    >>> print(result)
    Metric: Metric
    Point Estimate: 5.500
    Confidence Interval (95%): [4.235, 6.765]
    Estimator : ClassicalMeanEstimator
    n: 4
    """

    def _compute_mean_estimate(self, y: NDArray) -> float:
        mean = np.nanmean(y)
        return mean

    def _compute_std_estimate(self, y: NDArray) -> float:
        n_not_nan = np.sum(~np.isnan(y))
        std = np.nanstd(y, ddof=1) / np.sqrt(n_not_nan)
        return std

    def estimate(
        self,
        y: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
    ) -> ClassicalMeanInferenceResult:
        """Estimate the population mean using the classical sample mean.

        Parameters
        ----------
        y : NDArray
            Array of observations, shape ``(n_samples,)``.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval, e.g. ``0.95``
            for a 95 % CI. Defaults to ``0.95``.

        Returns
        -------
        ClassicalMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"ClassicalMeanEstimator"``), and ``n``
            (number of observations).
        """
        mean = self._compute_mean_estimate(y)
        std = self._compute_std_estimate(y)
        ci = CLTConfidenceInterval(
            mean=mean,
            std=std,
            confidence_level=confidence_level,
        )
        result = ClassicalMeanInferenceResult(
            confidence_interval=ci,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n=len(y),
        )
        return result
