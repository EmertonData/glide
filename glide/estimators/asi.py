from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
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
    >>> from glide.estimators.asi import ASIMeanEstimator
    >>> y_true = np.array([5.0, 6.0, np.nan, np.nan])
    >>> y_proxy = np.array([4.9, 6.1, 5.2, 6.1])
    >>> sampling_probabilities = np.array([0.5, 0.7, 0.6, 0.2])
    >>> estimator = ASIMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy, sampling_probabilities)
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
        sampling_probabilities: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        if not (len(y_true_all) == len(y_proxy) == len(sampling_probabilities)):
            raise ValueError("y_true, y_proxy, and sampling_probabilities must all have the same length")
        if np.min(sampling_probabilities) <= 0 or np.max(sampling_probabilities) > 1:
            raise ValueError("Sampling probabilities should be in (0, 1]")
        if np.isnan(y_proxy).any():
            raise ValueError("Input proxy values contain NaN")
        if len(np.unique(y_proxy)) == 1:
            raise ValueError("Input proxy values have zero variance")
        xi = (~np.isnan(y_true_all)).astype(float)
        y_true = np.where(np.isnan(y_true_all), -1.0, y_true_all)
        return y_true, y_proxy, xi, sampling_probabilities

    def _compute_lambda(
        self,
        y_data: Tuple[NDArray, NDArray, NDArray, NDArray],
        power_tuning: bool,
    ) -> float:
        if not power_tuning:
            return 1.0
        y_true, y_proxy, xi, pi = y_data
        a = y_proxy * (xi / pi - 1)
        b = y_true * xi / pi
        cov_matrix = np.cov(a, b, ddof=1)
        var, cov = cov_matrix[0]
        _lambda = cov / var
        return _lambda

    def _compute_rectified_labels(
        self,
        y_data: Tuple[NDArray, NDArray, NDArray, NDArray],
        _lambda: float,
    ) -> NDArray:
        y_true, y_proxy, xi, pi = y_data
        rectified_labels = _lambda * y_proxy + xi * (y_true - _lambda * y_proxy) / pi
        return rectified_labels

    def _compute_mean_estimate(
        self,
        rectified_labels: NDArray,
    ) -> float:
        mean_estimate = np.mean(rectified_labels)
        return mean_estimate

    def _compute_std_estimate(
        self,
        rectified_labels: NDArray,
    ) -> float:
        n = len(rectified_labels)
        std_estimate = np.std(rectified_labels, ddof=1) / np.sqrt(n)
        return std_estimate

    def estimate(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        sampling_probabilities: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        power_tuning: bool = True,
    ) -> SemiSupervisedMeanInferenceResult:
        """Estimate the population mean using Active Statistical Inference (ASI).

        Uses inverse-probability weighting (IPW) to correct for non-uniform sampling,
        combining labeled and unlabeled samples into a single IPW-corrected estimator.
        A power-tuning step (enabled by default) finds the λ that minimises asymptotic
        variance.

        Parameters
        ----------
        y_true : NDArray
            Array of shape ``(N,)`` with ground-truth labels. Use ``np.nan`` for
            unlabeled rows (ξ_i = 0); non-NaN rows are treated as labeled (ξ_i = 1).
        y_proxy : NDArray
            Array of shape ``(N,)`` with proxy predictions. Must be present for every
            row and must not contain NaN.
        sampling_probabilities : NDArray
            Array of shape ``(N,)`` with the pre-determined sampling probability
            π_i ∈ (0, 1] for each record.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        power_tuning : bool, optional
            If ``True`` (default), selects λ analytically to minimise asymptotic variance.
            If ``False``, uses λ = 1 (plain IPW estimator).

        Returns
        -------
        SemiSupervisedMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name, the estimator
            name (``"ASIMeanEstimator"``), and the counts ``n_true`` (labeled rows) and
            ``n_proxy`` (total rows).

        Raises
        ------
        ValueError
            If any value in ``sampling_probabilities`` is not in (0, 1], i.e.
            less than or equal to 0 or greater than 1.
        """
        y_data = self._preprocess(y_true, y_proxy, sampling_probabilities)
        _lambda = self._compute_lambda(y_data, power_tuning)
        rectified_labels = self._compute_rectified_labels(y_data, _lambda)
        mean_estimate = self._compute_mean_estimate(rectified_labels)
        std_estimate = self._compute_std_estimate(rectified_labels)

        y_true_processed, _, xi, _ = y_data
        n_true = int(xi.sum())
        n_proxy = len(y_proxy)

        confidence_interval = CLTConfidenceInterval(
            mean=mean_estimate, std=std_estimate, confidence_level=confidence_level
        )
        effective_sample_size = compute_effective_sample_size(y_true_processed[xi == 1], std_estimate)

        return SemiSupervisedMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_true,
            n_proxy=n_proxy,
            effective_sample_size=effective_sample_size,
        )
