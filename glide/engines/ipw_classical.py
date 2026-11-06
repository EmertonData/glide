from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import (
    _get_non_zero_mask,
    _validate_equal_lengths,
    _validate_label_prob_consistency,
    _validate_min_samples,
    _validate_probabilities,
)

IPWClassicalDataset = Tuple[NDArray, NDArray]


class IPWClassicalMeanEngine:
    def _preprocess(self, y: NDArray, sampling_probability: NDArray) -> IPWClassicalDataset:
        _validate_equal_lengths(y, sampling_probability, names=["y", "sampling_probability"])
        _validate_probabilities(sampling_probability)
        labeled_mask = ~np.isnan(y)
        _validate_label_prob_consistency(labeled_mask, sampling_probability)
        _validate_min_samples(y, "y")

        non_zero_probability_mask = _get_non_zero_mask(sampling_probability)
        y_filtered = y[non_zero_probability_mask]
        sampling_probability_filtered = sampling_probability[non_zero_probability_mask]
        return y_filtered, sampling_probability_filtered

    def _fit_tuning_parameter(self, dataset: IPWClassicalDataset, *, power_tuning: bool) -> None:
        return None

    def _compute_mean_and_std(self, dataset: IPWClassicalDataset, tuning_parameter: None) -> Tuple[float, float]:
        y, sampling_probability = dataset
        n_samples = len(sampling_probability)
        y_filled = np.nan_to_num(y, nan=0.0)
        ipw_weighted_values = y_filled / sampling_probability
        mean = np.mean(ipw_weighted_values)
        std = np.std(ipw_weighted_values, ddof=1) / np.sqrt(n_samples)
        return mean, std
