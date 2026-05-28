from math import floor
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.validation import (
    _get_non_zero_mask,
    _is_constant,
    _validate_equal_lengths,
    _validate_label_prob_consistency,
    _validate_probabilities,
    _validate_y_proxy,
)
from glide.estimators.ipw_classical import IPWClassicalMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult


class ASIMeanEstimator:
    """Estimator for population mean using Active Statistical Inference (ASI).

    This class implements the ASI method which extends PPI++ to non-uniform sampling.
    Each labeled sample has a known, pre-determined sampling probability π_i. Inverse
    probability weighting (IPW) corrects for this non-uniform selection, yielding valid
    confidence intervals under any sampling rule.

    The special case where all π_i are equal to n_labeled / n recovers PPI++ at λ = 1.

    References
    ----------
    Zrnic, Tijana, and Emmanuel J. Candès. "Active statistical inference." In Proceedings
    of the 41st International Conference on Machine Learning, pp. 62993-63010. 2024.

    Gligorić, Kristina, Tijana Zrnic, Cinoo Lee, Emmanuel Candes, and Dan Jurafsky.
    "Can unconfident llm annotations be used for confident conclusions?." In Proceedings
    of the 2025 Conference of the Nations of the Americas Chapter of the Association for
    Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),
    pp. 3514-3533. 2025.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators import ASIMeanEstimator
    >>> y_true = np.array([0.0, 1.0, np.nan, np.nan])
    >>> y_proxy = np.array([0.1, 0.9, 0.5, 0.5])
    >>> pi = np.array([0.8, 0.8, 0.8, 0.8])
    >>> estimator = ASIMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy, pi)
    >>> print(result)
    Metric: Metric
    Point Estimate: 0.548
    Confidence Interval (95%): [0.138, 0.958]
    Estimator : ASIMeanEstimator
    n_true: 2
    n_proxy: 4
    Effective Sample Size: 4
    """

    def _preprocess(
        self,
        y_true_all: NDArray,
        y_proxy: NDArray,
        pi: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        _validate_equal_lengths(y_true_all, y_proxy, pi, names=["y_true", "y_proxy", "pi"])
        _validate_y_proxy(y_proxy)
        _validate_probabilities(pi)
        y_true_non_nan_mask = ~np.isnan(y_true_all)
        _validate_label_prob_consistency(y_true_non_nan_mask, pi)
        xi = y_true_non_nan_mask.astype(float)

        non_zero_mask = _get_non_zero_mask(
            pi,
            "Some observations have pi=0. These will be excluded from the estimation.",
        )
        y_true_all = y_true_all[non_zero_mask]
        y_proxy = y_proxy[non_zero_mask]
        pi = pi[non_zero_mask]
        xi = xi[non_zero_mask]

        if _is_constant(y_proxy * (xi / pi - 1)):
            raise ValueError("'y_proxy' values lead to constant rectifiers.")

        y_true_filled = np.nan_to_num(y_true_all, nan=0)
        return y_true_filled, y_proxy, xi, pi

    def _compute_tuning_parameter(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        xi: NDArray,
        pi: NDArray,
        power_tuning: bool,
    ) -> float:
        if not power_tuning:
            return 1.0
        a = y_proxy * (xi / pi - 1)
        b = y_true * xi / pi
        cov_matrix = np.cov(a, b, ddof=1)
        var, cov = cov_matrix[0]
        _lambda = cov / var
        return _lambda

    def _compute_rectified_labels(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        xi: NDArray,
        pi: NDArray,
        _lambda: float,
    ) -> NDArray:
        rectified_labels = _lambda * y_proxy + xi * (y_true - _lambda * y_proxy) / pi
        return rectified_labels

    def estimate(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        pi: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        power_tuning: bool = True,
    ) -> PredictionPoweredMeanInferenceResult:
        """Estimate the population mean using Active Statistical Inference (ASI).

        Uses inverse-probability weighting (IPW) to correct for non-uniform sampling,
        combining labeled and unlabeled samples into a single IPW-corrected estimator.
        A power-tuning step (enabled by default) finds the λ that minimises asymptotic
        variance.

        Parameters
        ----------
        y_true : NDArray
            Array of shape ``(n_samples,)`` with ground-truth labels. Use ``np.nan`` for
            unlabeled samples (ξ_i = 0); non-NaN entries are treated as labeled (ξ_i = 1).
        y_proxy : NDArray
            Array of shape ``(n_samples,)`` with proxy predictions. Must be present for every
            sample and must not contain NaN.
        pi : NDArray
            Array of shape ``(n_samples,)`` with the pre-determined sampling probability
            π_i ∈ [0, 1] for each sample. Entries with π_i = 0 are excluded from all
            computations.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        power_tuning : bool, optional
            If ``True`` (default), selects λ analytically to minimise asymptotic variance.
            If ``False``, uses λ = 1 (plain IPW estimator).

        Returns
        -------
        PredictionPoweredMeanInferenceResult
            Contains a ``CLTConfidenceInterval``, metric name, estimator
            name (``"ASIMeanEstimator"``), and counts ``n_true`` (labeled samples) and
            ``n_proxy`` (total samples).

        Raises
        ------
        ValueError
            - If ``y_true``, ``y_proxy``, and ``pi`` do not all have the same length.
            - If any proxy value is NaN.
            - If the rectifiers ``y_proxy * (ξ_i / π_i - 1)`` are constant.
            - If any value in ``pi`` is not in [0, 1].
        """
        y_true_filled, y_proxy_filtered, xi, pi_filtered = self._preprocess(y_true, y_proxy, pi)

        n_true = int(xi.sum())
        n_proxy = len(pi_filtered)

        _lambda = self._compute_tuning_parameter(y_true_filled, y_proxy_filtered, xi, pi_filtered, power_tuning)
        rectified_labels = self._compute_rectified_labels(y_true_filled, y_proxy_filtered, xi, pi_filtered, _lambda)
        mean_estimate = np.mean(rectified_labels)
        std_estimate = np.std(rectified_labels, ddof=1) / np.sqrt(n_proxy)

        confidence_interval = CLTConfidenceInterval(
            mean=mean_estimate, std=std_estimate, confidence_level=confidence_level
        )
        classical_confidence_interval = IPWClassicalMeanEstimator().estimate(y_true, pi).confidence_interval
        effective_sample_size = floor(n_true * classical_confidence_interval.var / confidence_interval.var)

        return PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_true,
            n_proxy=n_proxy,
            effective_sample_size=effective_sample_size,
        )
