from typing import Dict, Hashable

import numpy as np
from numpy.typing import NDArray

from glide.core.utils import _split_labeled_unlabeled
from glide.core.validation import _validate_equal_lengths, _validate_has_no_nan, _validate_sample_sizes
from glide.engines.ppi import PPIDataset

StratifiedDataset = Dict[Hashable, PPIDataset]


def _preprocess(y_true: NDArray, y_proxy: NDArray, groups: NDArray) -> StratifiedDataset:
    _validate_has_no_nan(groups, "groups")
    _validate_equal_lengths(y_true, y_proxy, groups, names=["y_true", "y_proxy", "groups"])
    _validate_has_no_nan(y_proxy, "y_proxy")

    stratified_dataset = {}
    for stratum_id in np.unique(groups):
        stratum_mask = groups == stratum_id
        stratum_y_true, stratum_y_proxy = y_true[stratum_mask], y_proxy[stratum_mask]
        y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, labeled_mask = _split_labeled_unlabeled(
            stratum_y_true, stratum_y_proxy
        )
        _validate_sample_sizes(labeled_mask, stratum_id)
        stratified_dataset[stratum_id] = (y_true_labeled, y_proxy_labeled, y_proxy_unlabeled)
    return stratified_dataset
