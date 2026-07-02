from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import (
    _validate_equal_lengths,
    _validate_has_no_nan,
    _validate_sample_sizes,
    _validate_y_proxy,
    _validate_y_true,
)
from glide.estimators.core import _split_labeled_unlabeled


def _preprocess(
    y_true: NDArray,
    y_proxy: NDArray,
    groups: NDArray,
) -> List[Tuple[NDArray, NDArray, NDArray]]:
    _validate_has_no_nan(groups, "groups")
    _validate_equal_lengths(y_true, y_proxy, groups, names=["y_true", "y_proxy", "groups"])
    _validate_y_proxy(y_proxy)
    _validate_y_true(y_true)

    strata = []
    for stratum_id in np.unique(groups):
        stratum_mask = groups == stratum_id
        stratum_y_true = y_true[stratum_mask]
        stratum_y_proxy = y_proxy[stratum_mask]
        y_true_filtered, y_proxy_labeled, y_proxy_unlabeled, labeled_mask = _split_labeled_unlabeled(
            stratum_y_true, stratum_y_proxy
        )
        _validate_y_proxy(stratum_y_proxy, stratum_id)
        _validate_sample_sizes(labeled_mask, stratum_id)
        strata.append((y_true_filtered, y_proxy_labeled, y_proxy_unlabeled))

    return strata
