from typing import Tuple, Union, overload

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


@overload
def _scale_to_unit_risk(
    values: float, metric_lower_bound: float, metric_upper_bound: float, higher_is_better: bool
) -> float: ...
@overload
def _scale_to_unit_risk(
    values: NDArray, metric_lower_bound: float, metric_upper_bound: float, higher_is_better: bool
) -> NDArray: ...
def _scale_to_unit_risk(
    values: Union[float, NDArray],
    metric_lower_bound: float,
    metric_upper_bound: float,
    higher_is_better: bool,
) -> Union[float, NDArray]:
    scaled = (values - metric_lower_bound) / (metric_upper_bound - metric_lower_bound)
    if higher_is_better:
        scaled = 1.0 - scaled
    return scaled


@overload
def _scale_from_unit_risk(
    values: float, metric_lower_bound: float, metric_upper_bound: float, higher_is_better: bool
) -> float: ...
@overload
def _scale_from_unit_risk(
    values: NDArray, metric_lower_bound: float, metric_upper_bound: float, higher_is_better: bool
) -> NDArray: ...
def _scale_from_unit_risk(
    values: Union[float, NDArray],
    metric_lower_bound: float,
    metric_upper_bound: float,
    higher_is_better: bool,
) -> Union[float, NDArray]:
    if higher_is_better:
        values = 1.0 - values
    original = metric_lower_bound + values * (metric_upper_bound - metric_lower_bound)
    return original
