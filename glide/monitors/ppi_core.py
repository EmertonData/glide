from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import (
    _validate_bounds,
    _validate_equal_lengths,
    _validate_has_no_nan,
    _validate_non_empty,
    _validate_y_proxy,
    _validate_y_true,
)
from glide.estimators.core import _split_labeled_unlabeled
from glide.estimators.ppi_core import _compute_mean_estimate, _compute_std_estimate, _compute_tuning_parameter
from glide.monitors.core import _scale_to_unit_risk, _unique_ordered_batches


def _preprocess(
    y_true: NDArray,
    y_proxy: NDArray,
    batches: NDArray,
    higher_is_better: bool,
    threshold: float,
    confidence_level: float,
    metric_lower_bound: float,
    metric_upper_bound: float,
) -> Tuple[NDArray, NDArray, float, NDArray, NDArray, NDArray]:
    _validate_non_empty(y_true, "y_true")
    _validate_equal_lengths(y_true, y_proxy, batches, names=["y_true", "y_proxy", "batches"])
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
            f"'metric_upper_bound'={metric_upper_bound!r}; got {threshold!r}."
        ),
    )
    _validate_y_proxy(y_proxy)
    _validate_y_true(y_true)
    labeled_mask = ~np.isnan(y_true)
    labeled_values = y_true[labeled_mask]
    _validate_bounds(
        labeled_values,
        "y_true",
        lower=metric_lower_bound,
        upper=metric_upper_bound,
        error_message=(
            f"'y_true' values must lie between 'metric_lower_bound'={metric_lower_bound!r} and "
            f"'metric_upper_bound'={metric_upper_bound!r}; got values in "
            f"[{labeled_values.min()!r}, {labeled_values.max()!r}]."
        ),
    )
    _validate_bounds(
        y_proxy,
        "y_proxy",
        lower=metric_lower_bound,
        upper=metric_upper_bound,
        error_message=(
            f"'y_proxy' values must lie between 'metric_lower_bound'={metric_lower_bound!r} and "
            f"'metric_upper_bound'={metric_upper_bound!r}; got values in "
            f"[{y_proxy.min()!r}, {y_proxy.max()!r}]."
        ),
    )
    batch_identifiers, batch_codes = _unique_ordered_batches(batches)
    n_batches = len(batch_identifiers)
    batch_n_true = np.bincount(batch_codes[labeled_mask], minlength=n_batches)
    batch_n_proxy = np.bincount(batch_codes, minlength=n_batches)
    batch_n_unlabeled = batch_n_proxy - batch_n_true

    worst_labeled_position = np.argmin(batch_n_true)
    _validate_bounds(
        batch_n_true[worst_labeled_position],
        "y_true",
        lower=2,
        error_message=(
            f"'y_true' must have at least 2 labeled values per batch; got "
            f"{batch_n_true[worst_labeled_position]} in batch '{batch_identifiers[worst_labeled_position]}'."
        ),
    )
    worst_unlabeled_position = np.argmin(batch_n_unlabeled)
    _validate_bounds(
        batch_n_unlabeled[worst_unlabeled_position],
        "y_true",
        lower=2,
        error_message=(
            f"'y_true' must have at least 2 unlabeled values per batch; got "
            f"{batch_n_unlabeled[worst_unlabeled_position]} in batch "
            f"'{batch_identifiers[worst_unlabeled_position]}'."
        ),
    )

    risk_y_true = _scale_to_unit_risk(y_true, metric_lower_bound, metric_upper_bound, higher_is_better)
    risk_y_proxy = _scale_to_unit_risk(y_proxy, metric_lower_bound, metric_upper_bound, higher_is_better)
    risk_threshold = _scale_to_unit_risk(threshold, metric_lower_bound, metric_upper_bound, higher_is_better)
    return risk_y_true, risk_y_proxy, risk_threshold, batch_codes, batch_n_true, batch_n_proxy


def _compute_batch_estimates(
    risk_y_true: NDArray,
    risk_y_proxy: NDArray,
    batch_codes: NDArray,
    power_tuning: bool,
) -> Tuple[NDArray, NDArray]:
    n_batches = batch_codes[-1] + 1
    batch_mean_estimates = np.empty(n_batches)
    batch_std_estimates = np.empty(n_batches)
    for position in range(n_batches):
        if position == 0 or (not power_tuning):
            tuning_parameter = 1.0
        else:
            earlier_mask = batch_codes < position
            y_true_earlier, y_proxy_labeled_earlier, y_proxy_unlabeled_earlier, _ = _split_labeled_unlabeled(
                risk_y_true[earlier_mask], risk_y_proxy[earlier_mask]
            )
            tuning_parameter = _compute_tuning_parameter(
                y_true_earlier, y_proxy_labeled_earlier, y_proxy_unlabeled_earlier, power_tuning
            )

        batch_mask = batch_codes == position
        y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, _ = _split_labeled_unlabeled(
            risk_y_true[batch_mask], risk_y_proxy[batch_mask]
        )
        batch_mean_estimates[position] = _compute_mean_estimate(
            y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, tuning_parameter
        )
        batch_std_estimates[position] = _compute_std_estimate(
            y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, tuning_parameter
        )
    return batch_mean_estimates, batch_std_estimates
