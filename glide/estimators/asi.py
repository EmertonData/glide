from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.mean_inference_result import PredictionPoweredMeanInferenceResult
from glide.core.utils import compute_effective_sample_size


class ASIMeanEstimator:
    """Estimator for population mean using Active Statistical Inference (ASI).

    This class implements the ASI method which extends PPI++ to non-uniform sampling.
    Each labeled sample has a known, pre-determined sampling probability π_i. Inverse
    probability weighting (IPW) corrects for this non-uniform selection, yielding valid
    confidence intervals under any sampling rule.

    The special case where all π_i are equal to n_labeled / n recovers PPI++ at λ = 1.

    References
    ----------
    Zrnic, Tijana, and Emmanuel Candès. "Active statistical inference."
    arXiv:2403.03208 (2024). https://arxiv.org/abs/2403.03208

    Gligoric, Kristina, et al. "Confidence-driven inference."
    arXiv:2408.15204 (2024). https://arxiv.org/abs/2408.15204

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators import ASIMeanEstimator
    >>> y_true = np.array([5.0, 6.0, np.nan, np.nan])
    >>> y_proxy = np.array([4.9, 6.1, 5.2, 6.1])
    >>> pi = np.array([0.5, 0.7, 0.6, 0.2])
    >>> estimator = ASIMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy, pi)
    >>> print(result)
    Metric: Metric
    Point Estimate: 5.563
    Confidence Interval (95%): [5.084, 6.042]
    Estimator : ASIMeanEstimator
    n_true: 2
    n_proxy: 4
    Effective Sample Size: 8
    """

    def _preprocess(
        self,
        y_true_all: NDArray,
        y_proxy: NDArray,
        pi: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        if not (len(y_true_all) == len(y_proxy) == len(pi)):
            raise ValueError("y_true, y_proxy, and pi must all have the same length")
        if np.min(pi) <= 0 or np.max(pi) > 1:
            raise ValueError("Sampling probabilities should be in (0, 1]")
        if np.isnan(y_proxy).any():
            raise ValueError("Input proxy values contain NaN")
        if len(np.unique(y_proxy)) == 1:
            raise ValueError("Input proxy values have zero variance")

        y_true_non_nan_mask = ~np.isnan(y_true_all)
        xi = y_true_non_nan_mask.astype(float)

        if np.any(~y_true_non_nan_mask & (pi == 1)):
            raise ValueError("Samples with probability one of being labeled must be labeled")

        y_true = np.where(np.isnan(y_true_all), -1.0, y_true_all)
        return y_true, y_proxy, xi, pi

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
            π_i ∈ (0, 1] for each sample.
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
            - If all proxy values are identical (zero variance).
            - If any value in ``pi`` is not in (0, 1].
        """
        y_true_labeled, y_proxy, xi, pi = self._preprocess(y_true, y_proxy, pi)
        n_true = int(xi.sum())
        n_proxy = len(y_proxy)

        _lambda = self._compute_tuning_parameter(y_true_labeled, y_proxy, xi, pi, power_tuning)
        rectified_labels = self._compute_rectified_labels(y_true_labeled, y_proxy, xi, pi, _lambda)
        mean_estimate = np.mean(rectified_labels)
        std_estimate = np.std(rectified_labels, ddof=1) / np.sqrt(n_proxy)

        confidence_interval = CLTConfidenceInterval(
            mean=mean_estimate, std=std_estimate, confidence_level=confidence_level
        )
        effective_sample_size = compute_effective_sample_size(y_true_labeled[xi == 1], confidence_interval.var)

        return PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_true,
            n_proxy=n_proxy,
            effective_sample_size=effective_sample_size,
        )
