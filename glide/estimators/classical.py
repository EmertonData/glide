from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.engines.classical import ClassicalMeanEngine
from glide.mean_inference_results import ClassicalMeanInferenceResult


class ClassicalMeanEstimator:
    """Estimator for population mean using the classical sample mean.

    Uses only a single array ``y`` to compute the sample mean and its
    standard error via the Central Limit Theorem. This serves as a baseline
    that does not require proxy predictions.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators import ClassicalMeanEstimator
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

    def __init__(self) -> None:
        self._engine = ClassicalMeanEngine()

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

        Raises
        ------
        ValueError
            If ``y`` contains fewer than 2 non-NaN values.
        """
        dataset = self._engine.preprocess(y)
        n_samples = len(dataset)
        mean, std = self._engine.compute_mean_and_std(dataset, None)
        ci = CLTConfidenceInterval(
            mean=mean,
            std=std,
            confidence_level=confidence_level,
        )
        result = ClassicalMeanInferenceResult(
            confidence_interval=ci,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n=n_samples,
        )
        return result
