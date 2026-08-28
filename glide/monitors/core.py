from typing import List, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_bounds, _validate_equal_lengths, _validate_has_no_nan, _validate_non_empty


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


def _preprocess(
    fields: List[NDArray],
    field_names: List[str],
    batches: NDArray,
    higher_is_better: bool,
    confidence_level: float,
) -> Tuple[List[NDArray], NDArray, NDArray]:
    _validate_bounds(
        confidence_level,
        "confidence_level",
        lower=0.5,
        upper=1,
        left_inclusive=False,
        right_inclusive=False,
        error_message=f"'confidence_level' must be in (0.5, 1) for the asymptotic monitor; got {confidence_level!r}.",
    )
    _validate_non_empty(batches, "batches")
    _validate_equal_lengths(*fields, batches, names=[*field_names, "batches"])
    _validate_has_no_nan(batches, "batches")

    risk_fields = [_reorient(field, higher_is_better) for field in fields]
    batch_identifiers, batch_codes = _unique_ordered_batches(batches)
    return risk_fields, batch_identifiers, batch_codes


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
