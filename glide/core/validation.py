import warnings
from typing import Hashable, List, Optional

import numpy as np
from numpy.typing import NDArray


def _is_constant(array: NDArray) -> bool:
    return np.max(array) == np.min(array)


def _validate_has_no_nan(array: NDArray, name: str) -> None:
    if np.isnan(array).any():
        raise ValueError(f"'{name}' contains NaN values.")


def _get_non_zero_mask(values: NDArray, warning_message: Optional[str] = None) -> NDArray:
    non_zero_mask = values > 0
    if warning_message is not None and np.any(~non_zero_mask):
        warnings.warn(warning_message, UserWarning)
    return non_zero_mask


def _validate_y_proxy(y_proxy: NDArray, stratum_id: Optional[Hashable] = None) -> None:

    _validate_has_no_nan(y_proxy, "y_proxy")
    if _is_constant(y_proxy):
        stratum_part = f" in stratum '{stratum_id}'" if stratum_id is not None else ""
        raise ValueError(f"'y_proxy' values are constant{stratum_part}.")


def _validate_y_true(y_true: NDArray) -> None:
    labeled = y_true[~np.isnan(y_true)]
    if _is_constant(labeled):
        raise ValueError("'y_true' labeled values are constant.")


def _validate_uncertainties(uncertainties: NDArray) -> None:
    _validate_has_no_nan(uncertainties, "uncertainties")
    if np.any(uncertainties <= 0.0):
        raise ValueError("'uncertainties' must all be strictly positive; got a non-positive value.")


def _validate_probabilities(values: NDArray) -> None:
    if np.min(values) < 0 or np.max(values) > 1:
        raise ValueError("Sampling probabilities must be in [0, 1].")


def _validate_label_prob_consistency(labeled_mask: NDArray, pi: NDArray) -> None:
    if np.any(labeled_mask & (pi == 0)):
        raise ValueError("Samples with zero probability of being labeled cannot be labeled.")
    if np.any(~labeled_mask & (pi == 1)):
        raise ValueError("Samples with probability one of being labeled must be labeled.")


def _validate_equal_lengths(*arrays: NDArray, names: List[str]) -> None:
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        if len(names) == 2:
            names_str = f"'{names[0]}' and '{names[1]}'"
            lengths_str = f"{lengths[0]} and {lengths[1]}"
        else:
            names_str = ", ".join(f"'{n}'" for n in names[:-1]) + f", and '{names[-1]}'"
            lengths_str = ", ".join(str(length) for length in lengths[:-1]) + f", and {lengths[-1]}"
        raise ValueError(f"{names_str} must have the same length; got {lengths_str}.")


def _validate_burn_in_y_true(y_true: NDArray) -> None:
    if len(y_true) == 0:
        raise ValueError("'y_true' must be non-empty.")
    _validate_has_no_nan(y_true, "y_true")
    if _is_constant(y_true):
        raise ValueError("'y_true' label values are constant.")


def _validate_strictly_positive(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"'{name}' must be strictly positive; got {value}.")


def _validate_is_integer(param: int, name: str) -> None:
    if (not isinstance(param, (int, np.integer))) or isinstance(param, bool):
        raise ValueError(f"'{name}' must be an integer; got {param!r}.")


def _validate_budget_bound(budget: int, n_max: int) -> None:
    if budget > n_max:
        raise ValueError(
            f"'budget' must not exceed the number of samples; got budget={budget} but the dataset has {n_max} elements."
        )


def _validate_sample_sizes(
    labeled_mask: NDArray,
    stratum_id: Optional[Hashable] = None,
) -> None:
    n_labeled = labeled_mask.sum()
    n_unlabeled = (~labeled_mask).sum()
    if min(n_labeled, n_unlabeled) <= 1:
        stratum_part = f"stratum '{stratum_id}'" if stratum_id is not None else "dataset"
        raise ValueError(f"Too few labeled or unlabeled samples in {stratum_part}.")
