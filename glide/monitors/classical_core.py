from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_bounds, _validate_equal_lengths, _validate_has_no_nan, _validate_non_empty
from glide.monitors.core import _scale_to_unit_risk, _unique_ordered_batches


def _preprocess(
    y: NDArray,
    batches: NDArray,
    higher_is_better: bool,
    threshold: float,
    confidence_level: float,
    metric_lower_bound: float,
    metric_upper_bound: float,
) -> Tuple[NDArray, float, NDArray, NDArray]:
    _validate_non_empty(y, "y")
    _validate_equal_lengths(y, batches, names=["y", "batches"])
    _validate_has_no_nan(batches, "batches")
    _validate_bounds(
        confidence_level, "confidence_level", lower=0, upper=1, left_inclusive=False, right_inclusive=False
    )
    _validate_bounds(
        metric_lower_bound,
        "metric_lower_bound",
        upper=metric_upper_bound,
        right_inclusive=False,
        error_message=(
            f"'metric_lower_bound' must be strictly smaller than 'metric_upper_bound'; "
            f"got {metric_lower_bound!r} and {metric_upper_bound!r}."
        ),
    )
    _validate_bounds(
        threshold,
        "threshold",
        lower=metric_lower_bound,
        upper=metric_upper_bound,
        error_message=(
            f"'threshold' must lie between 'metric_lower_bound'={metric_lower_bound!r} and "
            f"'metric_upper_bound'={metric_upper_bound!r}."
        ),
    )
    labeled_mask = ~np.isnan(y)
    labeled_values = y[labeled_mask]
    _validate_bounds(
        labeled_values,
        "y",
        lower=metric_lower_bound,
        upper=metric_upper_bound,
        error_message=(
            f"'y' values must lie between 'metric_lower_bound'={metric_lower_bound!r} and "
            f"'metric_upper_bound'={metric_upper_bound!r}; got values in "
            f"[{labeled_values.min()!r}, {labeled_values.max()!r}]."
        ),
    )
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
    risk_y = _scale_to_unit_risk(labeled_values, metric_lower_bound, metric_upper_bound, higher_is_better)
    risk_threshold = _scale_to_unit_risk(threshold, metric_lower_bound, metric_upper_bound, higher_is_better)
    return risk_y, risk_threshold, batch_codes, batch_n


def _compute_batch_estimates(
    risk_y: NDArray,
    batch_codes: NDArray,
) -> Tuple[NDArray, NDArray]:
    batch_n = np.bincount(batch_codes)
    batch_sums = np.bincount(batch_codes, weights=risk_y)
    batch_mean_estimates = batch_sums / batch_n
    batch_squared_sums = np.bincount(batch_codes, weights=risk_y**2)
    batch_variances = np.maximum(batch_squared_sums - batch_n * batch_mean_estimates**2, 0.0) / (batch_n - 1)
    batch_std_estimates = np.sqrt(batch_variances / batch_n)
    return batch_mean_estimates, batch_std_estimates
