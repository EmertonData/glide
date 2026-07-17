from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_sequences.base import ConfidenceSequence
from glide.core.validation import _validate_bounds, _validate_equal_lengths, _validate_is_integer, _validate_non_empty


def _compute_asymptotic_bounds(
    batch_estimates: NDArray,
    batch_std_estimates: NDArray,
    miscoverage: float,
    tightest_at_batch: int,
) -> Tuple[NDArray, NDArray]:
    _validate_non_empty(batch_estimates, "batch_estimates")
    _validate_equal_lengths(batch_estimates, batch_std_estimates, names=["batch_estimates", "batch_std_estimates"])
    _validate_bounds(
        batch_std_estimates,
        "batch_std_estimates",
        lower=0.0,
        error_message=f"'batch_std_estimates' must be non-negative; got {batch_std_estimates.min()!r}.",
    )
    _validate_bounds(miscoverage, "miscoverage", lower=0.0, upper=0.5, left_inclusive=False, right_inclusive=False)
    _validate_is_integer(tightest_at_batch, "tightest_at_batch")
    _validate_bounds(tightest_at_batch, "tightest_at_batch", lower=1)
    n_batches = len(batch_estimates)
    batch_counts = np.arange(1, n_batches + 1)
    running_mean_estimates = np.cumsum(batch_estimates) / batch_counts

    variance_process = np.cumsum(batch_std_estimates**2)
    target_position = min(tightest_at_batch, n_batches) - 1
    target_intrinsic_time = variance_process[target_position]
    _validate_bounds(
        target_intrinsic_time,
        "batch_std_estimates",
        lower=0.0,
        left_inclusive=False,
        error_message=(
            f"'batch_std_estimates' must accumulate a positive variance by batch {target_position + 1}; "
            f"got {target_intrinsic_time!r}."
        ),
    )
    doubled_miscoverage = 2.0 * miscoverage
    log_term = -2.0 * np.log(doubled_miscoverage)
    tuning_scale_squared = (log_term + np.log(log_term + 1.0)) / target_intrinsic_time
    scaled_times = variance_process * tuning_scale_squared + 1.0
    boundary_widths = np.sqrt(
        2.0
        * scaled_times
        / (batch_counts**2 * tuning_scale_squared)
        * np.log(1.0 + np.sqrt(scaled_times) / doubled_miscoverage)
    )
    lower_bounds = running_mean_estimates - boundary_widths
    return running_mean_estimates, lower_bounds


@dataclass
class AsymptoticConfidenceSequence(ConfidenceSequence):
    """Anytime-valid asymptotic confidence sequence on a running mean.

    A Gaussian-mixture confidence sequence whose width scales with the known
    standard errors of the per-batch estimates instead of a worst-case bounded
    range, so the bounds stay tight when per-batch estimates are precise. The
    price is an asymptotic (rather than exact) time-uniform guarantee: per-batch
    estimates must be approximately Gaussian with consistently estimated
    variances.

    Parameters
    ----------
    running_mean_estimates : NDArray
        Per-look running mean of the per-batch estimates, in original metric units.
    confidence_bounds : NDArray
        Per-look harmful-side anytime-valid bound, in original metric units.

    References
    ----------
    Waudby-Smith, Ian, David Arbour, Ritwik Sinha, Edward H. Kennedy, and Aaditya
    Ramdas. "Time-uniform central limit theory and asymptotic confidence
    sequences." The Annals of Statistics 52, no. 6 (2024): 2613-2640.

    Robbins, Herbert. "Statistical methods related to the law of the iterated
    logarithm." The Annals of Mathematical Statistics 41, no. 5 (1970): 1397-1409.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.confidence_sequences import AsymptoticConfidenceSequence
    >>> sequence = AsymptoticConfidenceSequence(
    ...     running_mean_estimates=np.array([0.4, 0.6]),
    ...     confidence_bounds=np.array([0.1, 0.55]),
    ... )
    >>> sequence.test_null_hypothesis(0.5, alternative="larger")
    array([False,  True])
    """
