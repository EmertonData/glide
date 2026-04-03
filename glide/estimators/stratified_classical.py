from typing import Dict, Hashable

import numpy as np

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.dataset import Dataset
from glide.core.mean_inference_result import ClassicalMeanInferenceResult
from glide.estimators.classical import ClassicalMeanEstimator


class StratifiedClassicalMeanEstimator:
    """Stratified classical estimator for population mean.

    Extends mean estimation as in `ClassicalMeanEstimator` to datasets partitioned
    into strata (e.g. by language, domain, or data source). A per-stratum sample
    mean and standard error are computed independently, then combined with
    population-proportional weights:

        theta = sum_k  w_k * theta_k
        sigma2 = sum_k  w_k^2 * sigma2_k

    where ``w_k = n_k / n`` is the fraction of records in stratum *k*.

    It is assumed that ``w_k`` reflects the true weight of stratum *k* for all
    *k*.

    Examples
    --------
    >>> from glide.core.dataset import Dataset
    >>> from glide.estimators.stratified_classical import StratifiedClassicalMeanEstimator
    >>> records = [
    ...     {"y": 1.0, "group": "A"},
    ...     {"y": 3.0, "group": "A"},
    ...     {"y": 5.0, "group": "B"},
    ...     {"y": 7.0, "group": "B"},
    ... ]
    >>> dataset = Dataset(records)
    >>> estimator = StratifiedClassicalMeanEstimator()
    >>> result = estimator.estimate(dataset, y_field="y", groups_field="group")
    >>> print(result)
    Metric: Metric
    Point Estimate: 4.000
    Confidence Interval (95%): [2.614, 5.386]
    Estimator : StratifiedClassicalMeanEstimator
    n: 4
    """

    def __init__(self) -> None:
        self._classical_mean_estimator = ClassicalMeanEstimator()

    def _get_strata(self, dataset: Dataset, groups_field: str) -> Dict[Hashable, Dataset]:
        groups: Dict[Hashable, Dataset] = {}
        for record in dataset:
            group_id = record[groups_field]
            if group_id not in groups:
                groups[group_id] = Dataset()
            groups[group_id].append(record)
        return groups

    def estimate(
        self,
        dataset: Dataset,
        y_field: str,
        groups_field: str,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
    ) -> ClassicalMeanInferenceResult:
        """Estimate the population mean using stratified classical inference.

        Splits the dataset by ``groups_field``, computes a classical sample-mean
        estimate within each stratum, and combines them with
        population-proportional weights:

            theta = sum_k  w_k * theta_k
            sigma2 = sum_k  w_k^2 * sigma2_k

        where ``w_k = n_k / n`` is the fraction of records in stratum *k*.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing records with ``y_field`` and ``groups_field``
            columns. Every record must have both fields present.
        y_field : str
            Name of the column holding the observations.
        groups_field : str
            Name of the field whose unique values define the strata.
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

        Raises
        ------
        KeyError
            If any record is missing ``groups_field``.
        """
        strata = self._get_strata(dataset, groups_field)

        weighted_mean = 0.0
        weighted_var = 0.0

        for stratum_dataset in strata.values():
            w_k = len(stratum_dataset) / len(dataset)
            y = self._classical_mean_estimator._preprocess(stratum_dataset, y_field)
            mean_k = self._classical_mean_estimator._compute_mean_estimate(y)
            std_k = self._classical_mean_estimator._compute_std_estimate(y)
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
            n=len(dataset),
        )
        return result
