from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.dataset import Dataset
from glide.core.inference_result import InferenceResult


class PPIMeanEstimator:
    """Estimator for population mean using Prediction-Powered Inference (PPI).

    This class implements the PPI method which combines a small set of labeled samples
    with a large set of unlabeled samples whose labels are approximated by a proxy model.
    The method provides consistent estimates even when the proxy is imperfect.

    References
    ----------
    Angelopoulos, Anastasios N., Stephen Bates, Clara Fannjiang, Michael I. Jordan,
    and Tijana Zrnic. "Prediction-powered inference." Science 382, no. 6671 (2023):
    669-674.

    Examples
    --------
    >>> from glide.core.dataset import Dataset \n
    >>> from glide.estimators.ppi import PPIMeanEstimator \n
    >>> labeled = [{"y_true": 5.0, "y_proxy": 4.9}, {"y_true": 6.0, "y_proxy": 6.1}] \n
    >>> unlabeled = [{"y_proxy": 5.2}, {"y_proxy": 6.1}] \n
    >>> dataset = Dataset(labeled + unlabeled) \n
    >>> estimator = PPIMeanEstimator() \n
    >>> result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy") \n
    >>> print(result)
    Metric: Metric
    Point Estimate: 5.650
    Confidence Interval (95%): [4.75, 6.55]
    Estimator : PPIMeanEstimator
    n_true: 2
    n_proxy: 4
    """
    
    def _preprocess(self, dataset: Dataset, y_true_field: str, y_proxy_field: str) -> Tuple[NDArray, NDArray, NDArray]:
        data = dataset.to_numpy(fields=[y_true_field, y_proxy_field])
        y_true_all = data[:, 0]
        y_proxy_all = data[:, 1]
        labeled_mask = ~np.isnan(y_true_all)
        y_true = y_true_all[labeled_mask]
        y_proxy_labeled = y_proxy_all[labeled_mask]
        y_proxy_unlabeled = y_proxy_all[~labeled_mask]
        return y_true, y_proxy_labeled, y_proxy_unlabeled

    def _ppi_mean(self, y_data: Tuple[NDArray, NDArray, NDArray]) -> float:
        y_true, y_proxy_labeled, y_proxy_unlabeled = y_data
        rectifier = np.mean(y_true) - np.mean(y_proxy_labeled)
        proxy_mean = np.mean(y_proxy_unlabeled)
        ppi_mean = proxy_mean + rectifier
        return ppi_mean

    def _ppi_std(self, y_data: Tuple[NDArray, NDArray, NDArray]) -> float:
        y_true, y_proxy_labeled, y_proxy_unlabeled = y_data
        n = len(y_true)
        N = len(y_proxy_unlabeled)
        var = np.var(y_true - y_proxy_labeled, ddof=1) / n + np.var(y_proxy_unlabeled, ddof=1) / N
        ppi_std = np.sqrt(var)
        return ppi_std

    def estimate(
        self,
        dataset: Dataset,
        y_true_field: str,
        y_proxy_field: str,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
    ) -> InferenceResult:
        """Estimate the population mean using Prediction-Powered Inference (PPI).

        Combines a small set of labeled samples with a large set of unlabeled samples whose
        labels are approximated by a proxy (e.g. a pretrained model). The rectifier
        ``mean(y_true) - mean(y_proxy_labeled)`` corrects the bias of the
        proxy, yielding a consistent estimate even when the proxy is imperfect.

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

        Returns
        -------
        InferenceResult
            Contains the CLT-based confidence interval, the metric name,
            the estimator name (``"PPIMeanEstimator"``), and the counts
            ``n_true`` (labeled rows) and ``n_proxy`` (all rows with a proxy
            prediction).
        """
        y_true, y_proxy_labeled, y_proxy_unlabeled = self._preprocess(dataset, y_true_field, y_proxy_field)
        mean = self._ppi_mean(y_true, y_proxy_labeled, y_proxy_unlabeled)
        std = self._ppi_std(y_true, y_proxy_labeled, y_proxy_unlabeled)
        n = len(y_true)
        var_y_true = np.var(y_true, ddof=1)
        std_labeled = np.sqrt(var_y_true / n)
        ess = float(n * (std_labeled / std) ** 2) if std > 0 else None
        ci = CLTConfidenceInterval(
            mean=float(mean),
            std=float(std),
            confidence_level=confidence_level,
        )
        result = InferenceResult(
            result=ci,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n,
            n_proxy=len(y_proxy_unlabeled) + len(y_proxy_labeled),
            ess=ess,
        )
        return result
