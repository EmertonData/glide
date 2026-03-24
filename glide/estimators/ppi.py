from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.dataset import Dataset
from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
from glide.core.utils import compute_effective_sample_size


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
    >>> from glide.core.dataset import Dataset
    >>> from glide.estimators.ppi import PPIMeanEstimator
    >>> labeled = [{"y_true": 5.0, "y_proxy": 4.9}, {"y_true": 6.0, "y_proxy": 6.1}]
    >>> unlabeled = [{"y_proxy": 5.2}, {"y_proxy": 6.1}]
    >>> dataset = Dataset(labeled + unlabeled)
    >>> estimator = PPIMeanEstimator()
    >>> result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy")
    >>> print(result)
    Metric: Metric
    Point Estimate: 5.618
    Confidence Interval (95%): [4.92, 6.31]
    Estimator : PPIMeanEstimator
    n_true: 2
    n_proxy: 4
    Effective Sample Size: 3.0
    """

    def _preprocess(self, dataset: Dataset, y_true_field: str, y_proxy_field: str) -> Tuple[NDArray, NDArray, NDArray]:
        data = dataset.to_numpy(fields=[y_true_field, y_proxy_field])
        y_true_all = data[:, 0]
        y_proxy_all = data[:, 1]
        if np.isnan(y_proxy_all).any():
            raise ValueError("Input proxy values contain NaN")
        if len(np.unique(y_proxy_all)) == 1:
            raise ValueError("Input proxy values have zero variance")
        labeled_mask = ~np.isnan(y_true_all)
        y_true = y_true_all[labeled_mask]
        y_proxy_labeled = y_proxy_all[labeled_mask]
        y_proxy_unlabeled = y_proxy_all[~labeled_mask]
        return y_true, y_proxy_labeled, y_proxy_unlabeled

    def _compute_lambda(self, y_data: Tuple[NDArray, NDArray, NDArray], power_tuning: bool) -> float:
        if not power_tuning:
            return 1.0
        y_true, y_proxy_labeled, y_proxy_unlabeled = y_data
        n = len(y_true)
        N = len(y_proxy_unlabeled)
        y_proxy_all = np.hstack([y_proxy_labeled, y_proxy_unlabeled])
        cov = np.cov(y_true, y_proxy_labeled, ddof=1)[0, 1]
        var = np.var(y_proxy_all, ddof=1)
        _lambda = cov / ((1 + n / N) * var)
        return _lambda

    def _compute_mean_estimate(self, y_data: Tuple[NDArray, NDArray, NDArray], _lambda: float) -> float:
        y_true, y_proxy_labeled, y_proxy_unlabeled = y_data
        rectifier = np.mean(y_true) - _lambda * np.mean(y_proxy_labeled)
        proxy_mean = _lambda * np.mean(y_proxy_unlabeled)
        ppi_mean = proxy_mean + rectifier
        return ppi_mean

    def _compute_std_estimate(self, y_data: Tuple[NDArray, NDArray, NDArray], _lambda: float) -> float:
        y_true, y_proxy_labeled, y_proxy_unlabeled = y_data
        n = len(y_true)
        N = len(y_proxy_unlabeled)
        rectifier_var = np.var(y_true - _lambda * y_proxy_labeled, ddof=1) / n
        proxy_var = _lambda**2 * np.var(y_proxy_unlabeled, ddof=1) / N
        ppi_var = proxy_var + rectifier_var
        ppi_std = np.sqrt(ppi_var)
        return ppi_std

    def estimate(
        self,
        dataset: Dataset,
        y_true_field: str,
        y_proxy_field: str,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        power_tuning: bool = True,
    ) -> SemiSupervisedMeanInferenceResult:
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
        dataset : Dataset
            Dataset containing both labeled rows (where ``y_true_field`` is
            present) and unlabeled rows (where ``y_true_field`` is absent /
            NaN). ``y_proxy_field`` must be present for every row.
        y_true_field : str
            Name of the column holding ground-truth labels.
        y_proxy_field : str
            Name of the column holding proxy predictions.
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
        InferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"PPIMeanEstimator"``), and the counts
            ``n_true`` (labeled rows) and ``n_proxy`` (all rows with a proxy
            prediction).
        """
        y_data = self._preprocess(dataset, y_true_field, y_proxy_field)
        y_true, y_proxy_labeled, y_proxy_unlabeled = y_data
        _lambda = self._compute_lambda(y_data, power_tuning)
        mean = self._compute_mean_estimate(y_data, _lambda)
        std = self._compute_std_estimate(y_data, _lambda)
        effective_sample_size = compute_effective_sample_size(y_true, std)
        ci = CLTConfidenceInterval(
            mean=float(mean),
            std=float(std),
            confidence_level=confidence_level,
        )
        result = SemiSupervisedMeanInferenceResult(
            confidence_interval=ci,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=len(y_true),
            n_proxy=len(y_proxy_unlabeled) + len(y_proxy_labeled),
            effective_sample_size=effective_sample_size,
        )
        return result
