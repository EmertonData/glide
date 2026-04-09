from typing import Optional

import numpy as np
from numpy.typing import NDArray

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.mean_inference_result import ClassicalMeanInferenceResult


class IPWClassicalMeanEstimator:
    """Estimator for population mean using Inverse Probability Weighting (IPW).

    Extends the classical sample mean to handle non-uniform sampling.
    Each observation y_i is reweighted by 1/π_i, where π_i is the
    pre-determined probability that record i was selected for labeling.
    Some values of y_i may be NaN corresponding to unsampled instances.

    The ``budget`` parameter is optional and defaults to ``len(y)``. For the
    computation to be statistically valid, the sum of π_i should approxximately
    equal to ``budget``.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators.ipw_classical import IPWClassicalMeanEstimator
    >>> y = np.array([5.0, 6.0, 4.0, 7.0])
    >>> pi = np.array([0.5, 0.8, 0.4, 0.6])
    >>> estimator = IPWClassicalMeanEstimator()
    >>> result = estimator.estimate(y, pi)
    >>> print(result)
    Metric: Metric
    Point Estimate: 9.792
    Confidence Interval (95%): [8.108, 11.475]
    Estimator : IPWClassicalMeanEstimator
    n: 4
    """

    def _compute_mean_estimate(self, y: NDArray, sampling_probability: NDArray, budget: int) -> float:
        mean = np.nansum(y / sampling_probability) / budget
        return mean

    def _compute_std_estimate(self, y: NDArray, sampling_probability: NDArray, budget: int) -> float:
        mean_of_square = np.nansum((y**2) / sampling_probability) / (budget)
        square_of_mean = (np.nansum(y / sampling_probability) / budget) ** 2
        std = np.sqrt((mean_of_square - square_of_mean) * (budget / (budget - 1)))
        return std

    def estimate(
        self,
        y: NDArray,
        sampling_probability: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        budget: Optional[int] = None,
    ) -> ClassicalMeanInferenceResult:
        """Estimate the population mean using IPW-corrected sample mean.

        Parameters
        ----------
        y : NDArray
            1-D array of observations.
        sampling_probability : NDArray
            1-D array of pre-determined sampling probabilities π_i ∈ (0, 1],
            one per observation. Must have the same length as ``y``.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        budget : int, optional
            Normalisation constant B used as ``(1/B) * sum(y_i / π_i)``.
            Defaults to ``len(y)``.

        Returns
        -------
        ClassicalMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"IPWClassicalMeanEstimator"``), and ``n``
            (number of observations).
        """
        if np.min(sampling_probability) <= 0:
            raise ValueError(f"Minimum sampling probability should be > 0, got {np.min(sampling_probability)}")
        effective_budget = len(y) if budget is None else budget
        mean = self._compute_mean_estimate(y, sampling_probability, effective_budget)
        std = self._compute_std_estimate(y, sampling_probability, effective_budget)
        ci = CLTConfidenceInterval(mean=mean, std=std, confidence_level=confidence_level)
        result = ClassicalMeanInferenceResult(
            confidence_interval=ci,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n=len(y),
        )
        return result
