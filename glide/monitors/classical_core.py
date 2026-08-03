from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import (
    _validate_bounds,
    _validate_equal_lengths,
    _validate_has_no_nan,
    _validate_min_samples,
    _validate_non_empty,
)
from glide.monitors.core import _unique_ordered_batches


def _preprocess(
    y: NDArray,
    batches: NDArray,
    higher_is_better: bool,
    threshold: float,
    confidence_level: float,
) -> Tuple[NDArray, float, NDArray, NDArray]:
    _validate_non_empty(y, "y")
    _validate_equal_lengths(y, batches, names=["y", "batches"])
    _validate_has_no_nan(batches, "batches")
    _validate_bounds(
        confidence_level, "confidence_level", lower=0, upper=1, left_inclusive=False, right_inclusive=False
    )
    labeled_mask = ~np.isnan(y)
    labeled_values = y[labeled_mask]
    _validate_min_samples(labeled_values, "y")

    batches_labeled = batches[labeled_mask]
    batch_identifiers, batch_codes = _unique_ordered_batches(batches_labeled)
    batch_n = np.bincount(batch_codes)
    worst_batch_position = np.argmin(batch_n)
    _validate_bounds(
        batch_n[worst_batch_position],
        "y",
        lower=2,
        error_message=(
            f"'y' must have at least 2 non-NaN values per batch; got {batch_n[worst_batch_position]} "
            f"in batch '{batch_identifiers[worst_batch_position]}'."
        ),
    )
    if higher_is_better:
        sign = -1.0
    else:
        sign = 1.0
    risk_y = sign * labeled_values
    risk_threshold = sign * threshold
    return risk_y, risk_threshold, batch_codes, batch_n


def _compute_batch_estimates(
    risk_y: NDArray,
    batch_codes: NDArray,
) -> Tuple[NDArray, NDArray]:
    batch_n = np.bincount(batch_codes)
    batch_sums = np.bincount(batch_codes, weights=risk_y)
    batch_mean_estimates = batch_sums / batch_n
    batch_deviations = risk_y - batch_mean_estimates[batch_codes]
    batch_sum_squared_deviations = np.bincount(batch_codes, weights=batch_deviations**2)
    batch_variances = batch_sum_squared_deviations / (batch_n - 1)
    batch_std_estimates = np.sqrt(batch_variances / batch_n)
    return batch_mean_estimates, batch_std_estimates
