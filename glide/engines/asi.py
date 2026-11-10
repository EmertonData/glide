from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import (
    _get_non_zero_mask,
    _validate_equal_lengths,
    _validate_label_prob_consistency,
    _validate_non_constant,
    _validate_probabilities,
    _validate_sample_sizes,
    _validate_y_proxy,
)

ASIDataset = Tuple[NDArray, NDArray, NDArray]


class ASIMeanEngine:
    def _preprocess(self, y_true: NDArray, y_proxy: NDArray, sampling_probability: NDArray) -> ASIDataset:
        _validate_equal_lengths(
            y_true, y_proxy, sampling_probability, names=["y_true", "y_proxy", "sampling_probability"]
        )
        _validate_y_proxy(y_proxy)
        _validate_probabilities(sampling_probability)
        labeled_mask = ~np.isnan(y_true)
        _validate_label_prob_consistency(labeled_mask, sampling_probability)

        non_zero_probability_mask = _get_non_zero_mask(sampling_probability)
        y_true_filtered = y_true[non_zero_probability_mask]
        y_proxy_filtered = y_proxy[non_zero_probability_mask]
        sampling_probability_filtered = sampling_probability[non_zero_probability_mask]
        _validate_sample_sizes(~np.isnan(y_true_filtered))
        return y_true_filtered, y_proxy_filtered, sampling_probability_filtered

    def _fit_tuning_parameter(self, dataset: ASIDataset, *, power_tuning: bool) -> float:
        if not power_tuning:
            tuning_parameter = 1.0
            return tuning_parameter
        y_true, y_proxy, sampling_probability = dataset
        labeled_indicator = ~np.isnan(y_true)
        inverse_weights = labeled_indicator / sampling_probability
        proxy_term = y_proxy * (inverse_weights - 1)
        _validate_non_constant(proxy_term, "'y_proxy' values lead to constant rectifiers.")
        label_term = np.nan_to_num(y_true, nan=0.0) * inverse_weights
        variance, covariance = np.cov(proxy_term, label_term, ddof=1)[0]
        tuning_parameter = covariance / variance
        return tuning_parameter

    def _compute_mean_and_std(self, dataset: ASIDataset, tuning_parameter: float) -> Tuple[float, float]:
        y_true, y_proxy, sampling_probability = dataset
        labeled_indicator = ~np.isnan(y_true)
        n_samples = len(sampling_probability)
        proxy_projection = tuning_parameter * y_proxy
        residuals = np.nan_to_num(y_true, nan=0.0) - proxy_projection
        rectified_labels = proxy_projection + labeled_indicator * residuals / sampling_probability
        mean = np.mean(rectified_labels)
        std = np.std(rectified_labels, ddof=1) / np.sqrt(n_samples)
        return mean, std
