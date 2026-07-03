from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def _split_labeled_unlabeled(y_true: NDArray, y_proxy: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    labeled_mask = ~np.isnan(y_true)
    y_true_labeled = y_true[labeled_mask]
    y_proxy_labeled = y_proxy[labeled_mask]
    y_proxy_unlabeled = y_proxy[~labeled_mask]
    return y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, labeled_mask
