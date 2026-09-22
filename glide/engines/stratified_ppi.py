from typing import Dict, Hashable, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.utils import _split_labeled_unlabeled
from glide.core.validation import _validate_equal_lengths, _validate_has_no_nan, _validate_sample_sizes
from glide.engines.ppi import PPIDataset
from glide.engines.ppi_core import _compute_mean_estimate, _compute_std_estimate, _compute_tuning_parameter

StratifiedPPIDataset = Dict[Hashable, PPIDataset]
StratifiedTuningParameter = Dict[Hashable, float]


class StratifiedPPIMeanEngine:
    def preprocess(self, y_true: NDArray, y_proxy: NDArray, groups: NDArray) -> StratifiedPPIDataset:
        _validate_has_no_nan(groups, "groups")
        _validate_equal_lengths(y_true, y_proxy, groups, names=["y_true", "y_proxy", "groups"])

        stratified_dataset = {}
        for stratum_id in np.unique(groups):
            stratum_mask = groups == stratum_id
            stratum_y_true, stratum_y_proxy = y_true[stratum_mask], y_proxy[stratum_mask]
            _validate_has_no_nan(stratum_y_proxy, "y_proxy")
            y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, labeled_mask = _split_labeled_unlabeled(
                stratum_y_true, stratum_y_proxy
            )
            _validate_sample_sizes(labeled_mask, stratum_id)
            stratified_dataset[stratum_id] = (y_true_labeled, y_proxy_labeled, y_proxy_unlabeled)
        return stratified_dataset

    def fit_tuning_parameter(self, dataset: StratifiedPPIDataset, power_tuning: bool) -> StratifiedTuningParameter:
        tuning_parameter = {}
        for stratum_id, (y_true, y_proxy_labeled, y_proxy_unlabeled) in dataset.items():
            try:
                tuning_parameter[stratum_id] = _compute_tuning_parameter(
                    y_true, y_proxy_labeled, y_proxy_unlabeled, power_tuning
                )
            except ValueError as error:
                raise ValueError(f"{error} (stratum '{stratum_id}').") from error
        return tuning_parameter

    def compute_mean_and_std(
        self, dataset: StratifiedPPIDataset, tuning_parameter: StratifiedTuningParameter
    ) -> Tuple[float, float]:
        stratum_sizes = {
            stratum_id: len(y_true) + len(y_proxy_unlabeled)
            for stratum_id, (y_true, _, y_proxy_unlabeled) in dataset.items()
        }
        n_samples = sum(stratum_sizes.values())

        weighted_mean = 0.0
        weighted_var = 0.0
        for stratum_id, (y_true, y_proxy_labeled, y_proxy_unlabeled) in dataset.items():
            lambda_k = tuning_parameter.get(stratum_id, 1.0)
            mean_k = _compute_mean_estimate(y_true, y_proxy_labeled, y_proxy_unlabeled, lambda_k)
            std_k = _compute_std_estimate(y_true, y_proxy_labeled, y_proxy_unlabeled, lambda_k)
            w_k = stratum_sizes[stratum_id] / n_samples
            weighted_mean += w_k * mean_k
            weighted_var += w_k**2 * std_k**2

        std = np.sqrt(weighted_var)
        return weighted_mean, std
