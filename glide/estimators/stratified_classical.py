from typing import Dict, Hashable

import numpy as np
from numpy.typing import NDArray

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.mean_inference_result import ClassicalMeanInferenceResult


class StratifiedClassicalMeanEstimator:
    """Stratified classical estimator for population mean.

    Extends mean estimation as in `ClassicalMeanEstimator` to datasets partitioned
    into strata (e.g. by language, domain, or data source). A per-stratum sample
    mean and standard error are computed independently, then combined with
    population-proportional weights.

    This yields narrower confidence intervals than a flat classical estimate
    whenever strata differ in mean or variance, because variance is reduced by
    stratification.

    All per-stratum computations are delegated to an internal
    :class:`ClassicalMeanEstimator` instance — no formulas are duplicated.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators.stratified_classical import StratifiedClassicalMeanEstimator
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

    def _compute_mean_estimate(self, y: NDArray) -> float:
        mean = np.nanmean(y)
        return mean

    def _compute_std_estimate(self, y: NDArray) -> float:
        n_not_nan = np.sum(~np.isnan(y))
        std = np.nanstd(y, ddof=1) / np.sqrt(n_not_nan)
        return std

    def _get_strata(self, y: NDArray, groups: NDArray) -> Dict[Hashable, NDArray]:
        strata: Dict[Hashable, NDArray] = {}
        unique_groups = np.unique(groups)
        for group_id in unique_groups:
            mask = groups == group_id
            strata[group_id] = y[mask]
        return strata

    def estimate(
        self,
        y: NDArray,
        groups: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
    ) -> ClassicalMeanInferenceResult:
        """Estimate the population mean using stratified classical inference.

        Splits observations by ``groups``, computes a classical sample-mean
        estimate within each stratum, and combines them with
        Splits the data by unique values in ``groups``, computes a classical
        sample-mean estimate within each stratum, and combines them with
        population-proportional weights:

            theta = sum_k  w_k * theta_k
            sigma2 = sum_k  w_k^2 * sigma2_k

        where ``w_k = n_k / n`` is the fraction of records in stratum *k*.

        It is assumed that ``w_k`` reflects the true weight of stratum *k* for
        all *k*.

        Parameters
        ----------
        y : NDArray
            1-D array of observed values, one entry per record.
        groups : NDArray
            1-D array of group identifiers, parallel to ``y``. Unique values
            define the strata.
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

        Returns
        -------
        ClassicalMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"StratifiedClassicalMeanEstimator"``), and
            ``n`` (total number of records).
        """
        n_total = len(y)
        strata = self._get_strata(y, groups)

        weighted_mean = 0.0
        weighted_var = 0.0

        for y_stratum in strata.values():
            w_k = len(y_stratum) / n_total
            mean_k = self._compute_mean_estimate(y_stratum)
            std_k = self._compute_std_estimate(y_stratum)
            weighted_mean += w_k * mean_k
            weighted_var += w_k**2 * std_k**2

        std = np.sqrt(weighted_var)
        ci = CLTConfidenceInterval(
            mean=weighted_mean,
            std=std,
            confidence_level=confidence_level,
        )
        result = ClassicalMeanInferenceResult(
            confidence_interval=ci,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n=n_total,
        )
        return result
