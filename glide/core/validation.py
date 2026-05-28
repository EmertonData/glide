import warnings
from typing import Hashable, List, Optional

import numpy as np
from numpy.typing import NDArray


def _is_constant(array: NDArray) -> bool:
    return np.max(array) == np.min(array)


def _get_non_zero_pi_mask(pi: NDArray, warning_message: str) -> NDArray:
    non_zero_pi_mask = pi > 0
    if not np.all(non_zero_pi_mask):
        warnings.warn(warning_message, UserWarning)
    return non_zero_pi_mask


def _validate_y_proxy(y_proxy: NDArray, stratum_id: Optional[Hashable] = None) -> None:
    if np.isnan(y_proxy).any():
        raise ValueError("Input proxy values contain NaN")
    if _is_constant(y_proxy):
        if stratum_id is None:
            raise ValueError("Input proxy values have zero variance")
        raise ValueError(f"Input proxy values have zero variance in stratum '{stratum_id}'")


def _validate_y_true(y_true: NDArray) -> None:
    labeled = y_true[~np.isnan(y_true)]
    if _is_constant(labeled):
        raise ValueError("Labeled y_true values have zero variance")


def _validate_uncertainties(uncertainties: NDArray) -> None:
    if np.any(np.isnan(uncertainties)):
        raise ValueError(
            "All uncertainty values must be finite; got a NaN value. "
            "A NaN uncertainty score cannot be used to compute sampling probabilities."
        )
    if np.any(uncertainties <= 0.0):
        raise ValueError(
            "All uncertainty values must be strictly positive; got a non-positive value. "
            "An observation with zero or negative uncertainty would never be selected."
        )


def _validate_non_constant(values: NDArray) -> None:
    if _is_constant(values):
        raise ValueError("Input values lead to rectifiers with zero variance")


def _validate_sampling_probabilities(pi: NDArray) -> None:
    if np.min(pi) < 0 or np.max(pi) > 1:
        raise ValueError("Sampling probabilities should be in [0, 1]")


def _validate_pi_consistency(labeled_mask: NDArray, pi: NDArray) -> None:
    if np.any(labeled_mask & (pi == 0)):
        raise ValueError("Samples with non-zero probability of being labeled cannot be labeled")
    if np.any(~labeled_mask & (pi == 1)):
        raise ValueError("Samples with probability one of being labeled must be labeled")


def _validate_equal_lengths(*arrays: NDArray, names: List[str]) -> None:
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        if len(names) == 2:
            names_str = f"{names[0]} and {names[1]}"
            lengths_str = f"{lengths[0]} and {lengths[1]}"
        else:
            names_str = ", ".join(names[:-1]) + f", and {names[-1]}"
            lengths_str = ", ".join(str(length) for length in lengths[:-1]) + f", and {lengths[-1]}"
        raise ValueError(f"{names_str} must have the same length, got {lengths_str}")


def _validate_budget(budget: int, max_size: int) -> None:
    if (not isinstance(budget, (int, np.integer))) or isinstance(budget, bool) or budget <= 0:
        raise ValueError(f"'budget' must be a strictly positive integer; got {budget!r}.")
    if budget > max_size:
        raise ValueError(
            f"'budget' must not exceed the number of samples; "
            f"got budget={budget} but the dataset has {max_size} elements."
        )


def _validate_sample_sizes(
    labeled_mask: NDArray,
    stratum_id: Optional[Hashable] = None,
) -> None:
    n_labeled = labeled_mask.sum()
    n_unlabeled = (~labeled_mask).sum()
    if min(n_labeled, n_unlabeled) <= 1:
        if stratum_id is None:
            raise ValueError("Too few labeled or unlabeled samples in dataset")
        raise ValueError(f"Too few labeled or unlabeled samples in stratum '{stratum_id}'")
