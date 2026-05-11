from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray


def preprocess(
    y_true: NDArray,
    y_proxy: NDArray,
    groups: NDArray,
) -> List[Tuple[NDArray, NDArray, NDArray]]:
    if len(y_true) != len(y_proxy) or len(y_true) != len(groups):
        raise ValueError(
            f"y_true, y_proxy, and groups must have the same length, "
            f"got {len(y_true)}, {len(y_proxy)}, and {len(groups)}"
        )
    if np.isnan(y_proxy).any():
        raise ValueError("Input proxy values contain NaN")

    strata = []
    for stratum_id in np.unique(groups):
        stratum_mask = groups == stratum_id
        stratum_y_true = y_true[stratum_mask]
        stratum_y_proxy = y_proxy[stratum_mask]
        labeled_mask = ~np.isnan(stratum_y_true)
        n_labeled = labeled_mask.sum()
        n_unlabeled = stratum_mask.sum() - n_labeled
        if len(np.unique(stratum_y_proxy)) == 1:
            raise ValueError(f"Input proxy values have zero variance in stratum '{stratum_id}'")
        if min(n_labeled, n_unlabeled) <= 1:
            raise ValueError(f"Too few labeled or unlabeled samples in stratum '{stratum_id}'")

        y_true_labeled = stratum_y_true[labeled_mask]
        y_proxy_labeled = stratum_y_proxy[labeled_mask]
        y_proxy_unlabeled = stratum_y_proxy[~labeled_mask]
        strata.append((y_true_labeled, y_proxy_labeled, y_proxy_unlabeled))

    return strata
