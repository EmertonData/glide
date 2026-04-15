from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
from glide.core.utils import compute_effective_sample_size


class StratifiedPPIMeanEstimator:
    """Stratified PPI++ estimator for population mean.

    Extends Prediction-Powered Inference to datasets that are naturally partitioned
    into strata (e.g. by language, domain, or data source). A per-stratum power-tuned
    lambda is computed independently for each stratum, and the final estimate is a
    population-proportional weighted average of the per-stratum PPI++ estimates.

    This yields narrower confidence intervals than standard PPI++ whenever strata differ
    in proxy quality or relative size, because the optimal lambda can adapt to each
    stratum's signal-to-noise ratio.

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
    >>> import numpy as np
    >>> from glide.estimators.stratified_ppi import StratifiedPPIMeanEstimator
    >>> y_true = np.array([1.0, 2.0, np.nan, np.nan, 4.0, 5.0, np.nan, np.nan])
    >>> y_proxy = np.array([1.1, 2.2, 1.5, 1.8, 3.9, 5.1, 4.5, 4.8])
    >>> groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> estimator = StratifiedPPIMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy, groups)
    >>> print(result.estimator_name)
    StratifiedPPIMeanEstimator
    >>> print(result)
    Metric: Metric
    Point Estimate: 3.086
    Confidence Interval (95%): [2.720, 3.452]
    Estimator : StratifiedPPIMeanEstimator
    n_true: 4
    n_proxy: 8
    Effective Sample Size: 95
    """

    def _preprocess(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        groups: NDArray,
    ) -> List[Tuple[NDArray, NDArray, NDArray]]:
        if np.isnan(y_proxy).any():
            raise ValueError("Input proxy values contain NaN")

        strata = []
        for stratum_id in np.unique(groups):
            stratum_mask = groups == stratum_id
            stratum_y_true = y_true[stratum_mask]
            stratum_y_proxy = y_proxy[stratum_mask]
            labeled_mask = ~np.isnan(stratum_y_true)
            n_labeled = labeled_mask.sum()
            n_unlabeled = stratum_mask.sum() - n_labeled
            if min(n_labeled, n_unlabeled) <= 1:
                raise RuntimeError(f"Too few labeled or unlabeled samples in dataset stratum '{stratum_id}'")
            if len(np.unique(stratum_y_proxy)) == 1:
                raise ValueError(f"Input proxy values have zero variance in stratum '{stratum_id}'")

            y_true_labeled = stratum_y_true[labeled_mask]
            y_proxy_labeled = stratum_y_proxy[labeled_mask]
            y_proxy_unlabeled = stratum_y_proxy[~labeled_mask]
            strata.append((y_true_labeled, y_proxy_labeled, y_proxy_unlabeled))

        return strata

    def _compute_tuning_parameter(
        self,
        y_true: NDArray,
        y_proxy_labeled: NDArray,
        y_proxy_unlabeled: NDArray,
        power_tuning: bool,
    ) -> float:
        if not power_tuning:
            return 1.0
        n = len(y_true)
        N = len(y_proxy_unlabeled)
        y_proxy_all = np.hstack([y_proxy_labeled, y_proxy_unlabeled])
        cov = np.cov(y_true, y_proxy_labeled, ddof=1)[0, 1]
        var = np.var(y_proxy_all, ddof=1)
        _lambda = cov / ((1 + n / N) * var)
        return _lambda

    def _compute_mean_estimate(
        self,
        y_true: NDArray,
        y_proxy_labeled: NDArray,
        y_proxy_unlabeled: NDArray,
        _lambda: float,
    ) -> float:
        rectifier = np.mean(y_true) - _lambda * np.mean(y_proxy_labeled)
        proxy_mean = _lambda * np.mean(y_proxy_unlabeled)
        mean_estimate = proxy_mean + rectifier
        return mean_estimate

    def _compute_std_estimate(
        self,
        y_true: NDArray,
        y_proxy_labeled: NDArray,
        y_proxy_unlabeled: NDArray,
        _lambda: float,
    ) -> float:
        n = len(y_true)
        N = len(y_proxy_unlabeled)
        rectifier_var = np.var(y_true - _lambda * y_proxy_labeled, ddof=1) / n
        proxy_var = _lambda**2 * np.var(y_proxy_unlabeled, ddof=1) / N
        var_estimate = proxy_var + rectifier_var
        std_estimate = np.sqrt(var_estimate)
        return std_estimate

    def estimate(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        groups: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        power_tuning: bool = True,
    ) -> SemiSupervisedMeanInferenceResult:
        """Estimate the population mean using Stratified PPI++.

        Splits arrays by unique values in ``groups``, computes a power-tuned PPI++
        estimate within each stratum, and combines them with
        population-proportional weights:

            theta = sum_k  w_k * theta_k(lambda_k)
            sigma2 = sum_k  w_k^2 * sigma2_k(lambda_k)

        where ``w_k = (n_k + N_k) / (n + N)`` is the fraction of records in stratum *k*.

        Note that this assumes n_k / n and N_k / N are approximately the same for all k
        which is important for statistical validity.

        Labeled and unlabeled records are distinguished by ``NaN`` in ``y_true``:
        a record is labeled if its ``y_true`` entry is not ``NaN``.

        Parameters
        ----------
        y_true : NDArray
            Array of observations, shape ``(n_samples,)``.
            Labeled entries are finite; unlabeled entries are ``np.nan``.
        y_proxy : NDArray
            Array of proxy predictions, shape ``(n_samples,)``.
            Must be fully populated (no NaN). Must have nonzero variance.
        groups : NDArray
            Array of integer stratum identifiers, shape ``(n_samples,)``. Unique
            values define the strata.
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
        ValueError
            If any proxy value is NaN, or if all proxy values within a stratum are identical
            (zero variance), which would cause a division by zero when computing lambda.
        RuntimeError
            If any stratum has fewer than 2 labeled or fewer than 2 unlabeled records.
        """
        strata = self._preprocess(y_true, y_proxy, groups)

        weighted_mean = 0.0
        weighted_var = 0.0
        total_size = len(y_true)

        for y_true_labeled, y_proxy_labeled, y_proxy_unlabeled in strata:
            stratum_size = len(y_true_labeled) + len(y_proxy_unlabeled)
            w_k = stratum_size / total_size

            lambda_k = self._compute_tuning_parameter(y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, power_tuning)
            mean_k = self._compute_mean_estimate(y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, lambda_k)
            std_k = self._compute_std_estimate(y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, lambda_k)

            weighted_mean += w_k * mean_k
            weighted_var += w_k**2 * std_k**2

        std = np.sqrt(weighted_var)
        n_true = np.sum(~np.isnan(y_true))
        effective_sample_size = compute_effective_sample_size(y_true, std)

        confidence_interval = CLTConfidenceInterval(
            mean=weighted_mean,
            std=std,
            confidence_level=confidence_level,
        )
        result = SemiSupervisedMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_true,
            n_proxy=total_size,
            effective_sample_size=effective_sample_size,
        )
        return result
