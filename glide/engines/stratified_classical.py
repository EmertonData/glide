from typing import Dict, Hashable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_equal_lengths, _validate_has_no_nan, _validate_min_samples
from glide.engines.classical import ClassicalMeanEngine

StratifiedClassicalDataset = Dict[Hashable, Tuple[NDArray, float]]


class StratifiedClassicalMeanEngine:
    def __init__(self) -> None:
        self._classical_engine = ClassicalMeanEngine()

    def _preprocess(
        self, y: NDArray, groups: NDArray, stratum_weights: Optional[NDArray] = None
    ) -> StratifiedClassicalDataset:
        _validate_has_no_nan(groups, "groups")
        _validate_equal_lengths(y, groups, names=["y", "groups"])
        unique_strata, stratum_codes = np.unique(groups, return_inverse=True)
        if stratum_weights is None:
            stratum_weights = np.bincount(stratum_codes) / len(groups)
        not_nan_mask = ~np.isnan(y)
        stratified_dataset = {}
        for stratum_code, stratum_id in enumerate(unique_strata):
            stratum_mask = (stratum_codes == stratum_code) & not_nan_mask
            y_stratum = y[stratum_mask]
            _validate_min_samples(y_stratum, "y", stratum_id)
            stratified_dataset[stratum_id] = (y_stratum, stratum_weights[stratum_code])
        return stratified_dataset

    def _fit_tuning_parameter(self, dataset: StratifiedClassicalDataset, *, power_tuning: bool) -> None:
        return None

    def _compute_mean_and_std(self, dataset: StratifiedClassicalDataset, tuning_parameter: None) -> Tuple[float, float]:
        weighted_mean = 0.0
        weighted_var = 0.0
        for y_stratum, stratum_weight in dataset.values():
            mean_k, std_k = self._classical_engine._compute_mean_and_std(y_stratum, tuning_parameter)
            weighted_mean += stratum_weight * mean_k
            weighted_var += stratum_weight**2 * std_k**2
        std = np.sqrt(weighted_var)
        return weighted_mean, std
