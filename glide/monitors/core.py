from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def _unique_ordered_batches(batches: NDArray) -> Tuple[NDArray, NDArray]:
    block_starts = np.ones(len(batches), dtype=bool)
    block_starts[1:] = batches[1:] != batches[:-1]
    batch_identifiers = batches[block_starts]
    n_distinct_batches = len(batch_identifiers)
    if len(np.unique(batch_identifiers)) != n_distinct_batches:
        raise ValueError(
            "'batches' must be grouped into contiguous blocks ordered oldest first; "
            "found interleaved batches. Please sort the data by batch before calling detect."
        )
    batch_codes = np.cumsum(block_starts) - 1
    return batch_identifiers, batch_codes


def _postprocess(
    risk_running_means: NDArray,
    risk_confidence_bounds: NDArray,
    risk_batch_mean_estimates: NDArray,
    higher_is_better: bool,
) -> Tuple[NDArray, NDArray, NDArray]:
    if higher_is_better:
        sign = -1.0
    else:
        sign = 1.0
    running_means = sign * risk_running_means
    confidence_bounds = sign * risk_confidence_bounds
    batch_mean_estimates = sign * risk_batch_mean_estimates
    return running_means, confidence_bounds, batch_mean_estimates
