from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.dataset import Dataset
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
    >>> from glide.core.dataset import Dataset
    >>> from glide.estimators.asi import ASIMeanEstimator
    >>> labeled = [{"y_true": 5.0, "y_proxy": 4.9, "pi": 0.5}, {"y_true": 6.0, "y_proxy": 6.1, "pi": 0.7}]
    >>> unlabeled = [{"y_proxy": 5.2, "pi": 0.6}, {"y_proxy": 6.1, "pi": 0.2}]
    >>> dataset = Dataset(labeled + unlabeled)
    >>> estimator = ASIMeanEstimator()
    >>> result = estimator.estimate(
    ...      dataset,
    ...      y_true_field="y_true",
    ...      y_proxy_field="y_proxy",
    ...      sampling_probability_field="pi"
    ... )
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
        dataset: Dataset,
        y_true_field: str,
        y_proxy_field: str,
        sampling_probability_field: str,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        data = dataset.to_numpy(fields=[y_true_field, y_proxy_field, sampling_probability_field])
        y_true_all = data[:, 0]
        y_proxy = data[:, 1]
        pi = data[:, 2]
        if np.min(pi) <= 0:
            raise ValueError(f"Minimum annotation probability should be > 0, got {np.min(pi)}")
        if np.isnan(y_proxy).any():
            raise ValueError("Input proxy values contain NaN")
        if len(np.unique(y_proxy)) == 1:
            raise ValueError("Input proxy values have zero variance")
        xi = (~np.isnan(y_true_all)).astype(float)
        # replace NaN values in y_true_all by zero
        y_true = np.nan_to_num(y_true_all, nan=-1.0)
        return y_true, y_proxy, xi, pi

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
        _lambda = float(cov / var)
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
        mean_estimate = float(np.mean(rectified_labels))
        return mean_estimate

    def _compute_std_estimate(
        self,
        rectified_labels: NDArray,
    ) -> float:
        n = len(rectified_labels)
        std_estimate = float(np.std(rectified_labels, ddof=1) / np.sqrt(n))
        return std_estimate

    def estimate(
        self,
        dataset: Dataset,
        y_true_field: str,
        y_proxy_field: str,
        sampling_probability_field: str,
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
        dataset : Dataset
            Dataset where every record carries a proxy prediction and a sampling
            probability. Records that also have ``y_true_field`` are treated as
            labeled (ξ_i = 1); all others are unlabeled (ξ_i = 0).
        y_true_field : str
            Name of the column holding ground-truth labels (present only for labeled rows).
        y_proxy_field : str
            Name of the column holding proxy predictions (required for every row).
        sampling_probability_field : str
            Name of the column holding the pre-determined sampling probability π_i ∈ (0, 1]
            for each record. Mandatory — every record must carry this field.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        power_tuning : bool, optional
            If ``True`` (default), selects λ analytically to minimise asymptotic variance.
            If ``False``, uses λ = 1 (plain IPW estimator).

        Returns
        -------
        InferenceResult
            Contains the CLT-based confidence interval, the metric name, the estimator
            name (``"ASIMeanEstimator"``), and the counts ``n_true`` (labeled rows) and
            ``n_proxy`` (total rows).
        """
        y_data = self._preprocess(dataset, y_true_field, y_proxy_field, sampling_probability_field)
        _lambda = self._compute_lambda(y_data, power_tuning)
        rectified_labels = self._compute_rectified_labels(y_data, _lambda)
        mean_estimate = self._compute_mean_estimate(rectified_labels)
        std_estimate = self._compute_std_estimate(rectified_labels)

        y_true, y_proxy, xi, _ = y_data
        n_true = int(xi.sum())
        n_proxy = len(y_proxy)

        confidence_interval = CLTConfidenceInterval(
            mean=mean_estimate, std=std_estimate, confidence_level=confidence_level
        )
        effective_sample_size = compute_effective_sample_size(y_true[xi == 1], std_estimate)

        return SemiSupervisedMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_true,
            n_proxy=n_proxy,
            effective_sample_size=effective_sample_size,
        )
