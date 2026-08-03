from typing import Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray


@overload
def _reorient(value: float, higher_is_better: bool) -> float: ...
@overload
def _reorient(value: NDArray, higher_is_better: bool) -> NDArray: ...
def _reorient(value: Union[float, NDArray], higher_is_better: bool) -> Union[float, NDArray]:
    if higher_is_better:
        reoriented_value = -value
    else:
        reoriented_value = value
    return reoriented_value


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
    running_means = _reorient(risk_running_means, higher_is_better)
    confidence_bounds = _reorient(risk_confidence_bounds, higher_is_better)
    batch_mean_estimates = _reorient(risk_batch_mean_estimates, higher_is_better)
    return running_means, confidence_bounds, batch_mean_estimates
