from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import BootstrapConfidenceInterval
from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
from glide.core.utils import compute_effective_sample_size


class PTDMeanEstimator:
    """Estimator for population mean using Predict-Then-Debias (PTD).

    Combines a small set of labeled samples with a large set of unlabeled
    samples whose labels are approximated by a proxy model. Confidence
    intervals are constructed via a bootstrap percentile method, requiring
    no distributional assumptions on the proxy quality.

    The bootstrap uses the CLT-based Algorithm 3 from Kluger et al. (2025):
    the unlabeled proxy mean is computed once on the full unlabeled set and
    its sampling variability is simulated with a Gaussian draw at each
    iteration, making the per-iteration cost O(n_labeled) rather than
    O(n_labeled + n_unlabeled), where n_labeled and n_unlabeled are the
    number of labeled and unlabeled samples respectively.

    References
    ----------
    Kluger, Daniel M., et al. "Prediction-Powered Inference with Imputed
    Covariates and Nonuniform Sampling." arXiv:2501.18577 (2025).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators.ptd import PTDMeanEstimator
    >>> y_true = np.array([5.0, 6.0, np.nan, np.nan])
    >>> y_proxy = np.array([4.9, 6.1, 5.2, 6.1])
    >>> estimator = PTDMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy, n_bootstrap=5, random_seed=0)
    >>> print(result)
    Metric: Metric
    Point Estimate: 5.552
    Confidence Interval (95%): [5.211, 5.865]
    Estimator : PTDMeanEstimator
    n_true: 2
    n_proxy: 4
    Effective Sample Size: 5
    """

    def _preprocess(self, y_true_all: NDArray, y_proxy_all: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        if len(y_true_all) != len(y_proxy_all):
            raise ValueError(
                f"y_true and y_proxy must have the same length, got {len(y_true_all)} and {len(y_proxy_all)}"
            )
        if np.isnan(y_proxy_all).any():
            raise ValueError("Input proxy values contain NaN")
        if len(np.unique(y_proxy_all)) == 1:
            raise ValueError("Input proxy values have zero variance")
        labeled_mask = ~np.isnan(y_true_all)
        n_labeled = labeled_mask.sum()
        n_unlabeled = len(y_true_all) - n_labeled
        if min(n_labeled, n_unlabeled) <= 1:
            raise RuntimeError("Too few labeled or unlabeled samples in dataset")
        y_true = y_true_all[labeled_mask]
        y_proxy_labeled = y_proxy_all[labeled_mask]
        y_proxy_unlabeled = y_proxy_all[~labeled_mask]
        return y_true, y_proxy_labeled, y_proxy_unlabeled

    def _compute_unlabeled_proxy_mean(self, y_proxy_unlabeled: NDArray) -> float:
        mean_proxy_unlabeled = np.mean(y_proxy_unlabeled)
        return mean_proxy_unlabeled

    def _compute_unlabeled_proxy_var(self, y_proxy_unlabeled: NDArray) -> float:
        n_unlabeled = len(y_proxy_unlabeled)
        var_proxy_unlabeled = np.var(y_proxy_unlabeled, ddof=1) / n_unlabeled
        return var_proxy_unlabeled

    def _compute_bootstrap_labeled_estimates(
        self,
        y_true: NDArray,
        y_proxy_labeled: NDArray,
        n_bootstrap: int,
        rng: np.random.Generator,
    ) -> Tuple[NDArray, NDArray]:
        n_labeled = len(y_true)
        idx = rng.choice(n_labeled, size=(n_bootstrap, n_labeled), replace=True)
        y_true_means = np.mean(y_true[idx], axis=1)
        y_proxy_labeled_means = np.mean(y_proxy_labeled[idx], axis=1)
        return y_true_means, y_proxy_labeled_means

    def _compute_tuning_parameter(
        self,
        bootstraps: Tuple[NDArray, NDArray],
        var_proxy_unlabeled: float,
        power_tuning: bool,
    ) -> float:
        if not power_tuning:
            return 1.0
        y_true_means, y_proxy_labeled_means = bootstraps
        cov_matrix = np.cov(y_true_means, y_proxy_labeled_means, ddof=1)
        cov = cov_matrix[0, 1]
        var_proxy_labeled = cov_matrix[1, 1]
        denom = var_proxy_labeled + var_proxy_unlabeled
        _lambda = cov / denom
        return _lambda

    def _compute_bootstrap_mean_estimates(
        self,
        bootstraps: Tuple[NDArray, NDArray],
        mean_proxy_unlabeled: float,
        var_proxy_unlabeled: float,
        _lambda: float,
        rng: np.random.Generator,
    ) -> NDArray:
        y_true_means, y_proxy_labeled_means = bootstraps
        z = rng.standard_normal(len(y_true_means))
        unlabeled_means = mean_proxy_unlabeled + np.sqrt(var_proxy_unlabeled) * z
        rectifier_means = y_true_means - _lambda * y_proxy_labeled_means
        bootstrap_mean_estimates = _lambda * unlabeled_means + rectifier_means
        return bootstrap_mean_estimates

    def estimate(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        n_bootstrap: int = 2000,
        power_tuning: bool = True,
        random_seed: Optional[int] = None,
    ) -> SemiSupervisedMeanInferenceResult:
        """Estimate the population mean using Predict-Then-Debias (PTD).

        Combines a small set of labeled samples with a large set of unlabeled
        samples whose labels are approximated by a proxy model. The rectifier
        ``mean(y_true) - λ·mean(y_proxy_labeled)`` corrects the bias of the proxy,
        yielding a consistent estimate even when the proxy is imperfect.

        The tuning scalar λ and the confidence interval are both derived from a
        bootstrap over the labeled set only (Algorithm 3 from Kluger et al., 2025).
        The sampling variability of the unlabeled proxy mean is approximated by a
        single Gaussian draw per iteration, keeping the per-iteration cost O(n_labeled),
        where n_labeled is the number of labeled samples.

        Parameters
        ----------
        y_true : NDArray
            Array of labeled observations, shape ``(n_samples,)``.
            Labeled entries are finite; unlabeled entries are ``np.nan``.
        y_proxy : NDArray
            Array of proxy predictions, shape ``(n_samples,)``.
            Must be fully populated (no NaN). Must have nonzero variance.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        n_bootstrap : int, optional
            Number of bootstrap resamples. Defaults to ``2000``.
        power_tuning : bool, optional
            If ``True`` (default), estimate the optimal tuning scalar λ from
            the bootstrap covariances. If ``False``, use λ = 1.
        random_seed : int, optional
            Seed for the random number generator, for reproducibility.
            Defaults to ``None`` (non-deterministic).

        Returns
        -------
        SemiSupervisedMeanInferenceResult
            Contains a ``BootstrapConfidenceInterval``, metric name, estimator
            name (``"PTDMeanEstimator"``), and counts ``n_true`` / ``n_proxy``.
        """
        y_true_labeled, y_proxy_labeled, y_proxy_unlabeled = self._preprocess(y_true, y_proxy)
        n_labeled, n_unlabeled = len(y_true_labeled), len(y_proxy_unlabeled)
        rng = np.random.default_rng(random_seed)

        mean_proxy_unlabeled = self._compute_unlabeled_proxy_mean(y_proxy_unlabeled)
        var_proxy_unlabeled = self._compute_unlabeled_proxy_var(y_proxy_unlabeled)
        bootstraps = self._compute_bootstrap_labeled_estimates(y_true_labeled, y_proxy_labeled, n_bootstrap, rng)
        _lambda = self._compute_tuning_parameter(bootstraps, var_proxy_unlabeled, power_tuning)
        bootstrap_mean_estimates = self._compute_bootstrap_mean_estimates(
            bootstraps, mean_proxy_unlabeled, var_proxy_unlabeled, _lambda, rng
        )

        confidence_interval = BootstrapConfidenceInterval(
            bootstrap_estimates=bootstrap_mean_estimates,
            confidence_level=confidence_level,
        )
        effective_sample_size = compute_effective_sample_size(y_true_labeled, confidence_interval.var)
        result = SemiSupervisedMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_labeled,
            n_proxy=n_labeled + n_unlabeled,
            effective_sample_size=effective_sample_size,
        )
        return result
