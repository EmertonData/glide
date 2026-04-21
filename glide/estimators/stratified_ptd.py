from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import BootstrapConfidenceInterval
from glide.core.mean_inference_result import PredictionPoweredMeanInferenceResult
from glide.core.utils import compute_effective_sample_size
from glide.estimators.ptd_core import (
    _compute_ptd_bootstrap_labeled_means,
    _compute_ptd_bootstrap_mean_estimates,
    _compute_ptd_tuning_parameter,
)


class StratifiedPTDMeanEstimator:
    """Stratified Predict-Then-Debias estimator for population mean.

    Extends PTD to datasets partitioned into strata (e.g. by language, domain,
    or data source). A per-stratum power-tuned lambda is computed independently
    independently within each stratum, and the final confidence interval is
    constructed from a single bootstrap distribution obtained by combining the
    per-stratum bootstrap estimates with population-proportional weights.

    This yields narrower confidence intervals than standard PTD whenever strata
    differ in proxy quality, because the optimal lambda can adapt to each
    stratum's signal-to-noise ratio.

    Designed for the "small number of large strata" regime: the bootstrap CI
    becomes unreliable when strata are numerous and small (see Kluger et al., 2025,
    Appendix B.2). Note that the present implementation differs from Algorithm 6
    therein since it computes per-stratum tuning parameters. However, it remains
    statistically valid and may in fact be more precise.

    References
    ----------
    Kluger, Daniel M., et al. "Prediction-Powered Inference with Imputed
    Covariates and Nonuniform Sampling." arXiv:2501.18577 (2025).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators.stratified_ptd import StratifiedPTDMeanEstimator
    >>> y_true = np.array([5.0, 6.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan])
    >>> y_proxy = np.array([4.9, 6.1, 5.2, 6.1, 4.9, 6.1, 5.2, 6.1])
    >>> groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    >>> estimator = StratifiedPTDMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy, groups, n_bootstrap=5, random_seed=0)
    >>> print(result)
    Metric: Metric
    Point Estimate: 5.578
    Confidence Interval (95%): [5.400, 5.664]
    Estimator : StratifiedPTDMeanEstimator
    n_true: 4
    n_proxy: 8
    Effective Sample Size: 22
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
        for stratum_name in np.unique(groups):
            stratum_mask = groups == stratum_name
            stratum_y_true = y_true[stratum_mask]
            stratum_y_proxy = y_proxy[stratum_mask]

            labeled_mask = ~np.isnan(stratum_y_true)
            n_labeled = labeled_mask.sum()
            n_unlabeled = stratum_mask.sum() - n_labeled
            if min(n_labeled, n_unlabeled) <= 1:
                raise RuntimeError(f"Too few labeled or unlabeled samples in stratum '{stratum_name}'")
            if len(np.unique(stratum_y_proxy)) == 1:
                raise ValueError(f"Input proxy values have zero variance in stratum '{stratum_name}'")

            y_true_labeled = stratum_y_true[labeled_mask]
            y_proxy_labeled = stratum_y_proxy[labeled_mask]
            y_proxy_unlabeled = stratum_y_proxy[~labeled_mask]
            strata.append((y_true_labeled, y_proxy_labeled, y_proxy_unlabeled))

        return strata

    def estimate(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        groups: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        n_bootstrap: int = 2000,
        power_tuning: bool = True,
        random_seed: Optional[int] = None,
    ) -> PredictionPoweredMeanInferenceResult:
        """Estimate the population mean using Stratified Predict-Then-Debias.

        Splits arrays by unique values in ``groups``, applies the PTD bootstrap
        algorithm within each stratum with a per-stratum power-tuning, and
        combines the resulting per-stratum bootstrap arrays with
        population-proportional weights into a single ``BootstrapConfidenceInterval``:

            theta = sum_k  w_k * theta_k(lambda_k)

        where ``w_k`` is the fraction of samples in stratum *k*.

        Note that this assumes the portions of labeled vs unlabeled samples are
        approximately the same in all strata which is important for statistical
        validity.

        Labeled and unlabeled samples are distinguished by ``NaN`` in ``y_true``:
        a sample is labeled if its ``y_true`` entry is not ``NaN``.

        Parameters
        ----------
        y_true : NDArray
            Array of observations, shape ``(n_samples,)``.
            Labeled entries are finite; unlabeled entries are ``np.nan``.
        y_proxy : NDArray
            Array of proxy predictions, shape ``(n_samples,)``.
            Must be fully populated (no NaN). Must have nonzero variance within each stratum.
        groups : NDArray
            Array of stratum identifiers, shape ``(n_samples,)``. Unique values define the strata.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        n_bootstrap : int, optional
            Number of bootstrap resamples. Defaults to ``2000``.
        power_tuning : bool, optional
            If ``True`` (default), estimate the optimal per-stratum power-tuning parameter
            ``lambda_k`` from the bootstrap covariances. If ``False``, use ``lambda_k = 1.0``
            for all strata.
        random_seed : int, optional
            Seed for the random number generator, for reproducibility.
            Defaults to ``None`` (non-deterministic).

        Returns
        -------
        PredictionPoweredMeanInferenceResult
            Contains the bootstrap-based confidence interval, the metric name,
            the estimator name (``"StratifiedPTDMeanEstimator"``), and the counts
            ``n_true`` (total labeled rows) and ``n_proxy`` (total dataset size).

        Raises
        ------
        ValueError
            If any proxy value is NaN, or if all proxy values within a stratum are identical
            (zero variance), which would cause a division by zero when computing lambda.
        RuntimeError
            If any stratum has fewer than 2 labeled or fewer than 2 unlabeled samples.
        """
        strata = self._preprocess(y_true, y_proxy, groups)

        total_size = len(y_true)
        rng = np.random.default_rng(random_seed)

        combined_bootstrap_estimates = np.zeros(n_bootstrap)
        y_true_parts = []

        for y_true_labeled, y_proxy_labeled, y_proxy_unlabeled in strata:
            stratum_size = len(y_true_labeled) + len(y_proxy_unlabeled)
            w_k = stratum_size / total_size

            mean_proxy_unlabeled_k = np.mean(y_proxy_unlabeled)
            var_proxy_unlabeled_k = np.var(y_proxy_unlabeled, ddof=1) / len(y_proxy_unlabeled)

            bootstrap_y_true_means_k, bootstrap_y_proxy_labeled_means_k = _compute_ptd_bootstrap_labeled_means(
                y_true_labeled, y_proxy_labeled, n_bootstrap, rng
            )
            lambda_k = _compute_ptd_tuning_parameter(
                bootstrap_y_true_means_k, bootstrap_y_proxy_labeled_means_k, var_proxy_unlabeled_k, power_tuning
            )
            bootstrap_estimates_k = _compute_ptd_bootstrap_mean_estimates(
                bootstrap_y_true_means_k,
                bootstrap_y_proxy_labeled_means_k,
                mean_proxy_unlabeled_k,
                var_proxy_unlabeled_k,
                lambda_k,
                rng,
            )

            combined_bootstrap_estimates += w_k * bootstrap_estimates_k
            y_true_parts.append(y_true_labeled)

        confidence_interval = BootstrapConfidenceInterval(
            bootstrap_estimates=combined_bootstrap_estimates,
            confidence_level=confidence_level,
        )
        y_true_all = np.hstack(y_true_parts)
        effective_sample_size = compute_effective_sample_size(y_true_all, confidence_interval.var)
        result = PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=len(y_true_all),
            n_proxy=total_size,
            effective_sample_size=effective_sample_size,
        )
        return result
