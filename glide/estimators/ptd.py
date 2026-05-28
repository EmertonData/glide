from math import floor
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import BootstrapConfidenceInterval
from glide.core.validation import (
    _validate_equal_lengths,
    _validate_sample_sizes,
    _validate_y_proxy,
    _validate_y_true,
)
from glide.estimators.classical import ClassicalMeanEstimator
from glide.estimators.ptd_core import (
    _compute_bootstrap_labeled_means,
    _compute_bootstrap_mean_estimates,
    _compute_tuning_parameter,
)
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult


class PTDMeanEstimator:
    """Estimator for population mean using Predict-Then-Debias (PTD).

    Combines a small set of labeled samples with a large set of unlabeled
    samples whose labels are approximated by a proxy model. Confidence
    intervals are constructed via a bootstrap percentile method, requiring
    no distributional assumptions on the proxy quality.

    The bootstrap uses a CLT-based algorithm: the unlabeled proxy mean is
    computed once on the full unlabeled set and its sampling variability is
    simulated with a Gaussian draw at each iteration, making the per-iteration
    cost O(n_labeled) rather than O(n_labeled + n_unlabeled), where n_labeled
    and n_unlabeled are the number of labeled and unlabeled samples
    respectively.

    References
    ----------
    Kluger, Dan M., Kerri Lu, Tijana Zrnic, Sherrie Wang, and Stephen Bates.
    "Prediction-powered inference with imputed covariates and nonuniform sampling."
    arXiv preprint arXiv:2501.18577 (2025).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators import PTDMeanEstimator
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
        _validate_equal_lengths(y_true_all, y_proxy_all, names=["y_true", "y_proxy"])
        _validate_y_proxy(y_proxy_all)
        _validate_y_true(y_true_all)
        labeled_mask = ~np.isnan(y_true_all)
        _validate_sample_sizes(labeled_mask)
        y_true = y_true_all[labeled_mask]
        y_proxy_labeled = y_proxy_all[labeled_mask]
        y_proxy_unlabeled = y_proxy_all[~labeled_mask]
        return y_true, y_proxy_labeled, y_proxy_unlabeled

    def estimate(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        n_bootstrap: int = 2000,
        power_tuning: bool = True,
        random_seed: Optional[int] = None,
    ) -> PredictionPoweredMeanInferenceResult:
        """Estimate the population mean using Predict-Then-Debias (PTD).

        Combines a small set of labeled samples with a large set of unlabeled
        samples whose labels are approximated by a proxy model. The rectifier
        ``mean(y_true) - λ·mean(y_proxy_labeled)`` corrects the bias of the proxy,
        yielding a consistent estimate even when the proxy is imperfect.

        The tuning parameter λ and the confidence interval are both derived from a
        bootstrap over the labeled set only. The sampling variability of the
        unlabeled proxy mean is approximated by a single Gaussian draw per
        iteration, keeping the per-iteration cost O(n_labeled), where n_labeled
        is the number of labeled samples.

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
            If ``True`` (default), estimate the optimal tuning parameter λ from
            the bootstrap covariances. If ``False``, use λ = 1.
        random_seed : int, optional
            Seed for the random number generator, for reproducibility.
            Defaults to ``None`` (non-deterministic).

        Returns
        -------
        PredictionPoweredMeanInferenceResult
            Contains a ``BootstrapConfidenceInterval``, metric name, estimator
            name (``"PTDMeanEstimator"``), and counts ``n_true`` (labeled samples) and
            ``n_proxy`` (total samples).

        Raises
        ------
        ValueError
            - If ``y_true`` and ``y_proxy`` have different lengths.
            - If any proxy value is NaN.
            - If all proxy values are identical.
            - If there are fewer than 2 labeled or fewer than 2 unlabeled samples.
        """
        y_true_filtered, y_proxy_labeled, y_proxy_unlabeled = self._preprocess(y_true, y_proxy)
        n_labeled, n_unlabeled = len(y_true_filtered), len(y_proxy_unlabeled)
        rng = np.random.default_rng(random_seed)

        mean_proxy_unlabeled = np.mean(y_proxy_unlabeled)
        var_proxy_unlabeled = np.var(y_proxy_unlabeled, ddof=1) / n_unlabeled
        bootstrap_y_true_means, bootstrap_y_proxy_labeled_means = _compute_bootstrap_labeled_means(
            y_true_filtered, y_proxy_labeled, n_bootstrap, rng
        )
        lambda_ = _compute_tuning_parameter(
            bootstrap_y_true_means, bootstrap_y_proxy_labeled_means, var_proxy_unlabeled, power_tuning
        )
        bootstrap_mean_estimates = _compute_bootstrap_mean_estimates(
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
        classical_confidence_interval = ClassicalMeanEstimator().estimate(y_true_filtered).confidence_interval
        effective_sample_size = floor(n_labeled * classical_confidence_interval.var / confidence_interval.var)
        result = PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_labeled,
            n_proxy=n_labeled + n_unlabeled,
            effective_sample_size=effective_sample_size,
        )
        return result
