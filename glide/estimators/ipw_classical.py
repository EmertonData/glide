import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.mean_inference_result import ClassicalMeanInferenceResult


class IPWClassicalMeanEstimator:
    """Estimator for population mean using Inverse Probability Weighting (IPW).

    Extends the classical sample mean to handle non-uniform sampling.
    Each observation y_i is reweighted by 1/π_i, where π_i is the
    pre-determined probability that sample i was selected for labeling.
    Some values of y_i may be NaN corresponding to unsampled instances.

    For the computation to be statistically valid, the sum of π_i should be
    approximately equal to number of observed elements y_i.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators.ipw_classical import IPWClassicalMeanEstimator
    >>> y = np.array([5.0, 6.0, 4.0, np.nan, np.nan, np.nan])
    >>> pi = np.array([0.2, 0.8, 0.6, 0.6, 0.4, 0.4])
    >>> estimator = IPWClassicalMeanEstimator()
    >>> result = estimator.estimate(y, pi)
    >>> print(result)
    Metric: Metric
    Point Estimate: 6.528
    Confidence Interval (95%): [-1.230, 14.286]
    Estimator : IPWClassicalMeanEstimator
    n: 3
    """

    def estimate(
        self,
        y: NDArray,
        sampling_probability: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
    ) -> ClassicalMeanInferenceResult:
        """Estimate the population mean using IPW-corrected sample mean.

        Parameters
        ----------
        y : NDArray
            1-D array of observations, may contain unobserved NaN values.
        sampling_probability : NDArray
            1-D array of pre-determined sampling probabilities π_i ∈ (0, 1],
            one per observation. Must have the same length as ``y``.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.

        Returns
        -------
        ClassicalMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"IPWClassicalMeanEstimator"``), and ``n``
            (number of labeled observations).

        Raises
        ------
        ValueError
            If any value in ``sampling_probability`` is not in (0, 1], i.e.
            less than or equal to 0 or greater than 1.
        """
        if np.min(sampling_probability) <= 0 or np.max(sampling_probability) > 1:
            raise ValueError("Sampling probabilities should be in (0, 1]")

        n_labeled = np.sum(~np.isnan(y))
        total_size = len(y)
        ipw_weighted_values = np.nan_to_num(y, nan=0) / sampling_probability

        mean = np.mean(ipw_weighted_values)
        std = np.std(ipw_weighted_values, ddof=1) / np.sqrt(total_size)
        ci = CLTConfidenceInterval(mean=mean, std=std, confidence_level=confidence_level)
        result = ClassicalMeanInferenceResult(
            confidence_interval=ci,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n=n_labeled,
        )
        return result
