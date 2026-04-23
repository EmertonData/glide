from typing import Optional, Tuple

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


class IPWPTDMeanEstimator:
    """Estimator for population mean using IPW-corrected Predict-Then-Debias (IPW-PTD).

    Extends PTD to handle non-uniform, pre-determined labeled sampling probabilities
    via inverse probability weighting. The bootstrap percentile confidence interval
    requires no distributional assumptions on the proxy quality. The CLT speedup is
    applied to the unlabeled set; only the labeled set is resampled at each iteration.

    When all sampling probabilities are equal, recovers ``PTDMeanEstimator``. When n is
    large (CLT applies), produces inference equivalent to ``ASIMeanEstimator``, but without
    relying on the normal approximation for the labeled rectifier.

    References
    ----------
    Kluger, Daniel M., et al. "Prediction-Powered Inference with Imputed
    Covariates and Nonuniform Sampling." arXiv:2501.18577 (2025).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators.ipw_ptd import IPWPTDMeanEstimator
    >>> y_true  = np.array([1.0, 0.0, np.nan, np.nan])
    >>> y_proxy = np.array([0.9, 0.1, 0.8, 0.2])
    >>> pi      = np.array([0.4, 0.6, 0.3, 0.7])
    >>> estimator = IPWPTDMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy, pi, n_bootstrap=5, random_seed=0)
    >>> print(result)
    Metric: Metric
    Point Estimate: 0.253
    Confidence Interval (95%): [-0.082, 0.633]
    Estimator : IPWPTDMeanEstimator
    n_true: 2
    n_proxy: 4
    Effective Sample Size: 18
    """

    def _preprocess(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        pi: NDArray,
    ) -> Tuple[NDArray, NDArray, int, int]:
        if not (len(y_true) == len(y_proxy) == len(pi)):
            raise ValueError(
                f"y_true, y_proxy, and pi must have the same length, got {len(y_true)}, {len(y_proxy)}, and {len(pi)}"
            )
        if np.isnan(y_proxy).any():
            raise ValueError("Input proxy values contain NaN")
        if len(np.unique(y_proxy)) == 1:
            raise ValueError("Input proxy values have zero variance")
        xi = (~np.isnan(y_true)).astype(float)
        n_labeled = int(xi.sum())
        n_unlabeled = len(y_true) - n_labeled
        if min(n_labeled, n_unlabeled) <= 1:
            raise ValueError("Too few labeled or unlabeled samples in dataset")
        y_true_no_nan = np.nan_to_num(y_true, nan=0)
        if np.min(pi) <= 0 or np.max(pi) > 1:
            raise ValueError("Sampling probabilities should be in (0, 1]")
        return y_true_no_nan, xi, n_labeled, n_unlabeled

    def estimate(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        pi: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        n_bootstrap: int = 2000,
        power_tuning: bool = True,
        random_seed: Optional[int] = None,
    ) -> PredictionPoweredMeanInferenceResult:
        """Estimate the population mean using IPW-corrected Predict-Then-Debias.

        Labeled rows were drawn with known, non-uniform probabilities π_i. IPW
        reweights each resampled labeled observation by 1/π_i, restoring unbiasedness.
        The unlabeled proxy mean is not resampled: its sampling variability is injected
        via a single Gaussian draw per iteration (CLT speedup), keeping the per-iteration
        cost O(n_labeled) rather than O(n_labeled + n_unlabeled).

        Parameters
        ----------
        y_true : NDArray
            Ground-truth labels of shape ``(M,)``. Use ``np.nan`` for unlabeled rows.
        y_proxy : NDArray
            Proxy predictions of shape ``(M,)``. Must be finite for every row.
        pi : NDArray
            Pre-determined sampling probabilities of shape ``(M,)``. Must be > 0 for
            every labeled row (i.e. where ``y_true`` is not NaN). Values at unlabeled
            positions are ignored.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        n_bootstrap : int, optional
            Number of bootstrap resamples. Defaults to ``2000``.
        power_tuning : bool, optional
            If ``True`` (default), estimate the optimal tuning scalar λ from the
            bootstrap covariances. If ``False``, use λ = 1.
        random_seed : int, optional
            Seed for the random number generator, for reproducibility.
            Defaults to ``None`` (non-deterministic).

        Returns
        -------
        PredictionPoweredMeanInferenceResult
            Contains a ``BootstrapConfidenceInterval``, metric name, estimator name
            (``"IPWPTDMeanEstimator"``), and counts ``n_true`` / ``n_proxy``.

        Raises
        ------
        ValueError
            - If ``y_true``, ``y_proxy``, and ``pi`` do not all have the same length.
            - If any proxy value is NaN.
            - If all proxy values are identical (zero variance).
            - If there are fewer than 2 labeled or fewer than 2 unlabeled samples.
            - If any labeled sampling probability is ≤ 0.
        """
        y_true_clean, xi, n_labeled, n_unlabeled = self._preprocess(y_true, y_proxy, pi)
        rng = np.random.default_rng(random_seed)

        ipw_weighted_y_true_labeled = y_true_clean * xi / pi
        ipw_weighted_y_proxy_labeled = y_proxy * xi / pi
        ipw_weighted_y_proxy_unlabeled = y_proxy * (1 - xi) / (1 - pi)

        mean_proxy_unlabeled = np.mean(ipw_weighted_y_proxy_unlabeled)
        var_proxy_unlabeled = np.var(ipw_weighted_y_proxy_unlabeled, ddof=1) / len(ipw_weighted_y_proxy_unlabeled)

        bootstrap_y_true_means, bootstrap_y_proxy_labeled_means = _compute_ptd_bootstrap_labeled_means(
            ipw_weighted_y_true_labeled, ipw_weighted_y_proxy_labeled, n_bootstrap, rng
        )
        lambda_ = _compute_ptd_tuning_parameter(
            bootstrap_y_true_means, bootstrap_y_proxy_labeled_means, var_proxy_unlabeled, power_tuning
        )
        bootstrap_mean_estimates = _compute_ptd_bootstrap_mean_estimates(
            bootstrap_y_true_means,
            bootstrap_y_proxy_labeled_means,
            mean_proxy_unlabeled,
            var_proxy_unlabeled,
            lambda_,
            rng,
        )

        confidence_interval = BootstrapConfidenceInterval(
            bootstrap_estimates=bootstrap_mean_estimates,
            confidence_level=confidence_level,
        )
        effective_sample_size = compute_effective_sample_size(ipw_weighted_y_true_labeled, confidence_interval.var)
        result = PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_labeled,
            n_proxy=n_labeled + n_unlabeled,
            effective_sample_size=effective_sample_size,
        )
        return result
