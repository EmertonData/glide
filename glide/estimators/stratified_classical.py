from typing import Optional

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.engines.stratified_classical import StratifiedClassicalMeanEngine
from glide.mean_inference_results import ClassicalMeanInferenceResult


class StratifiedClassicalMeanEstimator:
    """Stratified classical estimator for population mean.

    Extends mean estimation as in `ClassicalMeanEstimator` to datasets partitioned
    into strata (e.g. by language, domain, or data source). A per-stratum sample
    mean and standard error are computed independently, then combined with
    population-proportional weights.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators import StratifiedClassicalMeanEstimator
    >>> y = np.array([1.0, 3.0, 5.0, 7.0])
    >>> groups = np.array(["A", "A", "B", "B"])
    >>> estimator = StratifiedClassicalMeanEstimator()
    >>> result = estimator.estimate(y, groups)
    >>> print(result)
    Metric: Metric
    Point Estimate: 4.000
    Confidence Interval (95%): [2.614, 5.386]
    Estimator : StratifiedClassicalMeanEstimator
    n: 4
    """

    def __init__(self) -> None:
        self._engine = StratifiedClassicalMeanEngine()

    def estimate(
        self,
        y: NDArray,
        groups: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        stratum_weights: Optional[NDArray] = None,
    ) -> ClassicalMeanInferenceResult:
        """Estimate the population mean using stratified classical inference.

        Splits observations by ``groups``, computes a classical sample-mean
        estimate within each stratum, and combines them with stratum weights:

            theta = sum_k  w_k * theta_k
            sigma2 = sum_k  w_k^2 * sigma2_k

        where ``w_k`` is the weight of stratum *k*. By default ``w_k`` is the
        share of all rows of ``groups`` falling in stratum *k*, whether or not
        their ``y`` entry is ``NaN``; pass ``stratum_weights`` to use a different
        weighting.

        It is assumed that ``w_k`` reflects the true weight of stratum *k* for
        all *k*.

        Parameters
        ----------
        y : NDArray
            Array of observations.
        groups : NDArray
            Array of group identifiers (same length as ``y``). Unique values
            define the strata.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval, e.g. ``0.95``
            for a 95 % CI. Defaults to ``0.95``.
        stratum_weights : NDArray, optional
            Stratum weights in sorted stratum order. When provided, these
            override the default weights, used as given with no normalisation.
            Defaults to ``None`` (each stratum's share of all rows, including
            those with a ``NaN`` entry in ``y``).

        Returns
        -------
        ClassicalMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"StratifiedClassicalMeanEstimator"``), and
            ``n`` (total number of samples).

        Raises
        ------
        ValueError
            - If ``groups`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``y`` and ``groups`` have different lengths.
            - If any stratum contains fewer than 2 non-NaN values.
        """
        stratified_dataset = self._engine._preprocess(y, groups, stratum_weights)
        weighted_mean, std = self._engine._compute_mean_and_std(stratified_dataset, None)
        n_labeled = np.sum(~np.isnan(y))
        ci = CLTConfidenceInterval(
            mean=weighted_mean,
            std=std,
            confidence_level=confidence_level,
        )
        result = ClassicalMeanInferenceResult(
            confidence_interval=ci,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n=n_labeled,
        )
        return result
