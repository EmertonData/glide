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
from glide.monitors.core import _unique_ordered_batches


def _preprocess(
    y_true: NDArray,
    y_proxy: NDArray,
    batches: NDArray,
    higher_is_better: bool,
    threshold: float,
    confidence_level: float,
) -> Tuple[NDArray, NDArray, float, NDArray, NDArray, NDArray]:
    _validate_non_empty(y_true, "y_true")
    _validate_equal_lengths(y_true, y_proxy, batches, names=["y_true", "y_proxy", "batches"])
    _validate_has_no_nan(batches, "batches")
    _validate_bounds(
        confidence_level, "confidence_level", lower=0, upper=1, left_inclusive=False, right_inclusive=False
    )
    _validate_y_proxy(y_proxy)
    _validate_y_true(y_true)
    labeled_mask = ~np.isnan(y_true)
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

    if higher_is_better:
        sign = -1.0
    else:
        sign = 1.0
    risk_y_true = sign * y_true
    risk_y_proxy = sign * y_proxy
    risk_threshold = sign * threshold
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
