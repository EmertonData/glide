from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_min_samples


class ClassicalMeanEngine:
    def preprocess(self, y: NDArray) -> NDArray:
        not_nan_mask = ~np.isnan(y)
        y_valid = y[not_nan_mask]
        _validate_min_samples(y_valid, "y")
        return y_valid

    def fit_tuning_parameter(self, dataset: NDArray, power_tuning: bool) -> None:
        return None

    def compute_mean_and_std(self, dataset: NDArray, tuning_parameter: Optional[float]) -> Tuple[float, float]:
        n_samples = len(dataset)
        mean = np.mean(dataset)
        std = np.std(dataset, ddof=1) / np.sqrt(n_samples)
        return mean, std
