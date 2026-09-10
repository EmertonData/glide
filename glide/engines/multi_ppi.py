from typing import Tuple

from numpy.typing import NDArray

from glide.core.utils import _split_labeled_unlabeled
from glide.core.validation import _validate_equal_lengths, _validate_has_no_nan, _validate_is_2d, _validate_sample_sizes
from glide.engines.multi_ppi_core import _compute_mean_estimate, _compute_std_estimate, _compute_tuning_parameter

MultiPPIDataset = Tuple[NDArray, NDArray, NDArray]


class MultiPPIMeanEngine:
    def preprocess(self, y_true: NDArray, y_proxies: NDArray) -> MultiPPIDataset:
        _validate_equal_lengths(y_true, y_proxies, names=["y_true", "y_proxies"])
        _validate_is_2d(y_proxies, "y_proxies")
        _validate_has_no_nan(y_proxies, "y_proxies")
        y_true_labeled, y_proxies_labeled, y_proxies_unlabeled, labeled_mask = _split_labeled_unlabeled(
            y_true, y_proxies
        )
        _validate_sample_sizes(labeled_mask)
        return y_true_labeled, y_proxies_labeled, y_proxies_unlabeled

    def fit_tuning_parameter(self, dataset: MultiPPIDataset, power_tuning: bool) -> NDArray:
        y_true, y_proxies_labeled, y_proxies_unlabeled = dataset
        tuning_parameter = _compute_tuning_parameter(y_true, y_proxies_labeled, y_proxies_unlabeled, power_tuning)
        return tuning_parameter

    def compute_mean_and_std(self, dataset: MultiPPIDataset, tuning_parameter: NDArray) -> Tuple[float, float]:
        y_true, y_proxies_labeled, y_proxies_unlabeled = dataset
        mean = _compute_mean_estimate(y_true, y_proxies_labeled, y_proxies_unlabeled, tuning_parameter)
        std = _compute_std_estimate(y_true, y_proxies_labeled, y_proxies_unlabeled, tuning_parameter)
        return mean, std
