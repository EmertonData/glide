from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.mean_inference_result import PredictionPoweredMeanInferenceResult
from glide.core.utils import compute_effective_sample_size
from glide.estimators.ppi_core import (
    _compute_ppi_mean_estimate,
    _compute_ppi_std_estimate,
    _compute_ppi_tuning_parameter,
)


class PPIMeanEstimator:
    """Estimator for population mean using Prediction-Powered Inference (PPI).

    This class implements the PPI method which combines a small set of labeled samples
    with a large set of unlabeled samples whose labels are approximated by a proxy model.
    The method provides consistent estimates even when the proxy is imperfect. An optional
    power-tuning mode (enabled by default) applies the optimal weight λ from PPI++,
    ensuring the confidence interval is never wider than the one obtained without the proxy.

    References
    ----------
    Angelopoulos, Anastasios N., Stephen Bates, Clara Fannjiang, Michael I. Jordan,
    and Tijana Zrnic. "Prediction-powered inference." Science 382, no. 6671 (2023):
    669-674.

    Angelopoulos, Anastasios N., John C. Duchi, and Tijana Zrnic. "Ppi++: Efficient
    prediction-powered inference." arXiv preprint arXiv:2311.01453 (2023).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators.ppi import PPIMeanEstimator
    >>> y_true = np.array([5.0, 6.0, np.nan, np.nan])
    >>> y_proxy = np.array([4.9, 6.1, 5.2, 6.1])
    >>> estimator = PPIMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy)
    >>> print(result)
    Metric: Metric
    Point Estimate: 5.618
    Confidence Interval (95%): [4.923, 6.312]
    Estimator : PPIMeanEstimator
    n_true: 2
    n_proxy: 4
    Effective Sample Size: 3
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
        n_labeled = sum(labeled_mask)
        n_unlabeled = len(y_true_all) - n_labeled
        if min(n_labeled, n_unlabeled) <= 1:
            raise ValueError("Too few labeled or unlabeled samples in dataset")
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
        power_tuning: bool = True,
    ) -> PredictionPoweredMeanInferenceResult:
        """Estimate the population mean using Prediction-Powered Inference (PPI).

        Combines a small set of labeled samples with a large set of unlabeled samples whose
        labels are approximated by a proxy (e.g. a pretrained model). The rectifier
        ``mean(y_true) - λ·mean(y_proxy_labeled)`` corrects the bias of the proxy, yielding
        a consistent estimate even when the proxy is imperfect.

        The weight λ interpolates between relying only on ``y_true`` (λ = 0) and the
        standard PPI estimate that leverages both ``y_true`` ``y_proxy`` with equal weights (λ = 1).
        When ``power_tuning=True`` (default), the optimal λ is computed via the PPI++
        closed-form formula to minimise the confidence interval width. When
        ``power_tuning=False``, λ = 1 and the estimator reduces to the classic PPI estimator.

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
            Target coverage for the confidence interval, e.g. ``0.95``
            for a 95 % CI. Defaults to ``0.95``.
        power_tuning : bool, optional
            If ``True`` (default), compute the optimal λ via the PPI++ formula
            to minimise CI width. If ``False``, use λ = 1 (classic PPI).

        Returns
        -------
        PredictionPoweredMeanInferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"PPIMeanEstimator"``), and the counts
            ``n_true`` (labeled observations) and ``n_proxy`` (all observations
            with a proxy prediction).

        Raises
        ------
        ValueError
            - If ``y_true`` and ``y_proxy`` have different lengths.
            - If any proxy value is NaN.
            - If all proxy values are identical (zero variance).
            - If there are fewer than 2 labeled or fewer than 2 unlabeled samples.
        """
        y_true_labeled, y_proxy_labeled, y_proxy_unlabeled = self._preprocess(y_true, y_proxy)
        n_labeled, n_unlabeled = len(y_true_labeled), len(y_proxy_unlabeled)
        lambda_ = _compute_ppi_tuning_parameter(y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, power_tuning)
        mean = _compute_ppi_mean_estimate(y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, lambda_)
        std = _compute_ppi_std_estimate(y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, lambda_)
        confidence_interval = CLTConfidenceInterval(
            mean=mean,
            std=std,
            confidence_level=confidence_level,
        )
        effective_sample_size = compute_effective_sample_size(y_true_labeled, confidence_interval.var)
        result = PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_labeled,
            n_proxy=n_labeled + n_unlabeled,
            effective_sample_size=effective_sample_size,
        )
        return result
