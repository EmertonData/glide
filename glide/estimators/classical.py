import numpy as np

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.dataset import Dataset
from glide.core.inference_result import InferenceResult
from glide.core.utils import compute_effective_sample_size


class ClassicalMeanEstimator:
    """Estimator for population mean using the classical sample mean.

    Uses only a single labeled field ``y`` to compute the sample mean and its
    standard error via the Central Limit Theorem. This serves as a baseline
    that does not require proxy predictions.

    Examples
    --------
    >>> from glide.core.dataset import Dataset \n
    >>> from glide.estimators.classical import ClassicalMeanEstimator \n
    >>> dataset = Dataset([{"y": 5.0}, {"y": 6.0}, {"y": 4.0}, {"y": 7.0}]) \n
    >>> estimator = ClassicalMeanEstimator() \n
    >>> result = estimator.estimate(dataset, y_field="y") \n
    >>> print(result)
    Metric: Metric
    Point Estimate: 5.500
    Confidence Interval (95%): [4.23, 6.77]
    Estimator : ClassicalMeanEstimator
    n_true: 4
    n_proxy: 0
    Effective Sample Size: 4.0
    """

    def estimate(
        self,
        dataset: Dataset,
        y_field: str,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
    ) -> InferenceResult:
        """Estimate the population mean using the classical sample mean.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing records with a ``y_field`` column.
        y_field : str
            Name of the column holding the observations.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval, e.g. ``0.95``
            for a 95 % CI. Defaults to ``0.95``.

        Returns
        -------
        InferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"ClassicalMeanEstimator"``), ``n_true``
            (number of observations), and ``n_proxy=0``.
        """
        data = dataset.to_numpy(fields=[y_field])[:, 0]
        y = data[~np.isnan(data)]
        mean = np.mean(y)
        std = np.std(y, ddof=1) / np.sqrt(len(y))
        effective_sample_size = compute_effective_sample_size(y, std)
        ci = CLTConfidenceInterval(
            mean=float(mean),
            std=float(std),
            confidence_level=confidence_level,
        )
        result = InferenceResult(
            result=ci,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=len(y),
            n_proxy=0,
            effective_sample_size=effective_sample_size,
        )
        return result
