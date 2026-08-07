from typing import Tuple

from numpy.typing import NDArray

from glide.core.utils import _split_labeled_unlabeled
from glide.core.validation import _validate_equal_lengths, _validate_has_no_nan, _validate_sample_sizes
from glide.engines.ppi_core import _compute_mean_estimate, _compute_std_estimate, _compute_tuning_parameter

PPIDataset = Tuple[NDArray, NDArray, NDArray]


class PPIMeanEngine:
    def preprocess(self, y_true: NDArray, y_proxy: NDArray) -> PPIDataset:
        _validate_equal_lengths(y_true, y_proxy, names=["y_true", "y_proxy"])
        _validate_has_no_nan(y_proxy, "y_proxy")
        y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, labeled_mask = _split_labeled_unlabeled(y_true, y_proxy)
        _validate_sample_sizes(labeled_mask)
        return y_true_labeled, y_proxy_labeled, y_proxy_unlabeled

    def fit_tuning_parameter(self, dataset: PPIDataset, power_tuning: bool) -> float:
        y_true, y_proxy_labeled, y_proxy_unlabeled = dataset
        tuning_parameter = _compute_tuning_parameter(y_true, y_proxy_labeled, y_proxy_unlabeled, power_tuning)
        return tuning_parameter

    def compute_mean_and_std(self, dataset: PPIDataset, tuning_parameter: float) -> Tuple[float, float]:
        y_true, y_proxy_labeled, y_proxy_unlabeled = dataset
        mean = _compute_mean_estimate(y_true, y_proxy_labeled, y_proxy_unlabeled, tuning_parameter)
        std = _compute_std_estimate(y_true, y_proxy_labeled, y_proxy_unlabeled, tuning_parameter)
        return mean, std
