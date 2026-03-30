from typing import Dict, Hashable

import numpy as np

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.dataset import Dataset
from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
from glide.core.utils import compute_effective_sample_size
from glide.estimators.ppi import PPIMeanEstimator


class StratifiedPPIMeanEstimator:
    """Stratified PPI++ estimator for population mean.

    Extends Prediction-Powered Inference to datasets that are naturally partitioned
    into strata (e.g. by language, domain, or data source). A per-stratum power-tuned
    lambda is computed independently for each stratum, and the final estimate is a
    population-proportional weighted average of the per-stratum PPI++ estimates.

    This yields narrower confidence intervals than standard PPI++ whenever strata differ
    in proxy quality or relative size, because the optimal lambda can adapt to each
    stratum's signal-to-noise ratio.

    All per-stratum computations are delegated to an internal :class:`PPIMeanEstimator`
    instance — no formulas are duplicated.

    References
    ----------
    Fisch, Adam, Joshua Maynez, R. Hofer, Bhuwan Dhingra, Amir Globerson, and
    William W. Cohen. "Stratified prediction-powered inference for effective hybrid
    evaluation of language models." Advances in Neural Information Processing
    Systems 37 (2024): 111489-111514.

    Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona. "A framework
    for efficient model evaluation through stratification, sampling, and estimation."
    In European Conference on Computer Vision, pp. 140-158. Cham: Springer Nature
    Switzerland, 2024.

    Examples
    --------
    >>> from glide.core.dataset import Dataset
    >>> from glide.estimators.stratified_ppi import StratifiedPPIMeanEstimator
    >>> records = [
    ...     {"y_true": 1.0, "y_proxy": 1.1, "group": "A"},
    ...     {"y_true": 2.0, "y_proxy": 2.2, "group": "A"},
    ...     {"y_proxy": 1.5, "group": "A"},
    ...     {"y_proxy": 1.8, "group": "A"},
    ...     {"y_true": 4.0, "y_proxy": 3.9, "group": "B"},
    ...     {"y_true": 5.0, "y_proxy": 5.1, "group": "B"},
    ...     {"y_proxy": 4.5, "group": "B"},
    ...     {"y_proxy": 4.8, "group": "B"},
    ... ]
    >>> dataset = Dataset(records)
    >>> estimator = StratifiedPPIMeanEstimator()
    >>> result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", groups_field="group")
    >>> print(result.estimator_name)
    StratifiedPPIMeanEstimator
    >>> print(result)
    Metric: Metric
    Point Estimate: 3.086
    Confidence Interval (95%): [2.72, 3.45]
    Estimator : StratifiedPPIMeanEstimator
    n_true: 4
    n_proxy: 8
    Effective Sample Size: 95.0
    """

    def __init__(self) -> None:
        self._ppi_mean_estimator = PPIMeanEstimator()

    def _get_strata(self, dataset: Dataset, groups_field: str) -> Dict[Hashable, Dataset]:
        """Split a dataset into per-stratum sub-datasets.

        Parameters
        ----------
        dataset : Dataset
            Full dataset; every record must contain ``groups_field``.
        groups_field : str
            Field whose unique values define the strata.

        Returns
        -------
        Dict[Hashable, Dataset]
            Mapping from each unique group value to the sub-dataset of records
            belonging to that stratum.

        Raises
        ------
        KeyError
            If any record is missing ``groups_field``.
        """
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
        y_true_field: str,
        y_proxy_field: str,
        groups_field: str,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        power_tuning: bool = True,
    ) -> SemiSupervisedMeanInferenceResult:
        """Estimate the population mean using Stratified PPI++.

        Splits the dataset by ``groups_field``, computes a power-tuned PPI++
        estimate within each stratum, and combines them with
        population-proportional weights::

            theta = sum_k  w_k * theta_k(lambda_k)
            sigma2 = sum_k  w_k^2 * sigma2_k(lambda_k)

        where ``w_k = N_k / N`` is the fraction of records in stratum *k*.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing both labeled rows (``y_true_field`` present) and
            unlabeled rows (``y_true_field`` absent/NaN). Every record must
            have ``y_proxy_field`` and ``groups_field``.
        y_true_field : str
            Name of the column holding ground-truth labels.
        y_proxy_field : str
            Name of the column holding proxy predictions.
        groups_field : str
            Name of the field whose unique values define the strata.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        power_tuning : bool, optional
            If ``True`` (default), compute the optimal ``lambda_k`` per stratum
            via the PPI++ formula. If ``False``, use ``lambda_k = 1.0`` for all
            strata (classic PPI).

        Returns
        -------
        SemiSupervisedMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"StratifiedPPIMeanEstimator"``), and the counts
            ``n_true`` (total labeled rows) and ``n_proxy`` (total dataset size).

        Raises
        ------
        KeyError
            If any record is missing ``groups_field``.
        """
        strata = self._get_strata(dataset, groups_field)
        N_total = len(dataset)

        weighted_mean = 0.0
        weighted_var = 0.0
        y_true_parts = []

        for stratum_name, stratum_dataset in strata.items():
            n_labeled = sum((y_true_field in record) for record in stratum_dataset)
            # at least 2 labeled and unlabeled samples are needed to compute a variance downstream
            N_k = len(stratum_dataset)
            if min(n_labeled, N_k - n_labeled) <= 1:
                raise RuntimeError(f"Too few labeled or unlabeled samples in stratum {stratum_name}")
            w_k = N_k / N_total
            y_data = self._ppi_mean_estimator._preprocess(stratum_dataset, y_true_field, y_proxy_field)
            lambda_k = self._ppi_mean_estimator._compute_lambda(y_data, power_tuning)
            mean_k = self._ppi_mean_estimator._compute_mean_estimate(y_data, lambda_k)
            std_k = self._ppi_mean_estimator._compute_std_estimate(y_data, lambda_k)

            weighted_mean += w_k * mean_k
            weighted_var += w_k**2 * std_k**2
            y_true_parts.append(y_data[0])

        std = np.sqrt(weighted_var)
        y_true_all = np.concatenate(y_true_parts)
        effective_sample_size = compute_effective_sample_size(y_true_all, std)

        confidence_interval = CLTConfidenceInterval(
            mean=weighted_mean,
            std=std,
            confidence_level=confidence_level,
        )
        return SemiSupervisedMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=len(y_true_all),
            n_proxy=N_total,
            effective_sample_size=effective_sample_size,
        )
