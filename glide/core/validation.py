import warnings
from typing import Hashable, List, Optional, Union

import numpy as np
from numpy.typing import NDArray


def _validate_non_constant(array: NDArray, error_message: str) -> None:
    if np.max(array) == np.min(array):
        raise ValueError(error_message)


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
    stratum_part = f" in stratum '{stratum_id}'" if stratum_id is not None else ""
    _validate_non_constant(y_proxy, f"'y_proxy' values are constant{stratum_part}.")


def _validate_y_true(y_true: NDArray, stratum_id: Optional[Hashable] = None) -> None:
    labeled = y_true[~np.isnan(y_true)]
    if len(labeled) == 0:
        raise ValueError("'y_true' contains only NaN values.")
    stratum_part = f" in stratum '{stratum_id}'" if stratum_id is not None else ""
    _validate_non_constant(labeled, f"'y_true' labeled values are constant{stratum_part}.")


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


def _validate_y_true_burn_in(y_true: NDArray) -> None:
    _validate_non_empty(y_true, "y_true")
    _validate_has_no_nan(y_true, "y_true")
    _validate_non_constant(y_true, "'y_true' label values are constant.")


def _validate_non_empty(array: Union[List, NDArray], name: str) -> None:
    if len(array) == 0:
        raise ValueError(f"'{name}' must be non-empty.")


def _validate_bounds(
    value: Union[float, NDArray],
    name: str,
    lower: float = -np.inf,
    upper: float = np.inf,
    left_inclusive: bool = True,
    right_inclusive: bool = True,
    error_message: Optional[str] = None,
) -> None:
    lower_valid = bool(np.all(value >= lower) if left_inclusive else np.all(value > lower))
    upper_valid = bool(np.all(value <= upper) if right_inclusive else np.all(value < upper))
    if not lower_valid or not upper_valid:
        if error_message is None:
            left_bracket = "[" if left_inclusive else "("
            right_bracket = "]" if right_inclusive else ")"
            interval_str = f"{left_bracket}{lower}, {upper}{right_bracket}"
            message = f"'{name}' must be in {interval_str}; got {value!r}."
        else:
            message = error_message
        raise ValueError(message)


def _validate_uncertainties(uncertainties: NDArray) -> None:
    _validate_has_no_nan(uncertainties, "uncertainties")
    _validate_bounds(
        uncertainties,
        "uncertainties",
        lower=0,
        left_inclusive=False,
        error_message="'uncertainties' must all be strictly positive; got a non-positive value.",
    )


def _validate_strictly_positive(value: float, name: str) -> None:
    _validate_bounds(
        value, name, lower=0, left_inclusive=False, error_message=f"'{name}' must be strictly positive; got {value}."
    )


def _validate_probabilities(values: NDArray) -> None:
    _validate_bounds(values, "Probabilities", lower=0, upper=1, error_message="Probabilities must be in [0, 1].")


def _validate_budget_bound(budget: int, n_max: int) -> None:
    _validate_bounds(
        budget,
        "budget",
        upper=n_max,
        error_message=f"'budget' must not exceed the number of samples; got budget={budget} but the dataset "
        f"has {n_max} elements.",
    )


def _validate_literal(arg: str, name: str, allowed: List[str]) -> None:
    if arg not in allowed:
        allowed_str = ", ".join(f"'{v}'" for v in allowed[:-1]) + f", or '{allowed[-1]}'"
        raise ValueError(f"'{name}' must be {allowed_str}; got {arg!r}.")


def _validate_is_integer(param: int, name: str) -> None:
    if (not isinstance(param, (int, np.integer))) or isinstance(param, bool):
        raise ValueError(f"'{name}' must be an integer; got {param!r}.")


def _validate_sample_sizes(
    labeled_mask: NDArray,
    stratum_id: Optional[Hashable] = None,
) -> None:
    n_labeled = labeled_mask.sum()
    n_unlabeled = (~labeled_mask).sum()
    if min(n_labeled, n_unlabeled) <= 1:
        stratum_part = f"stratum '{stratum_id}'" if stratum_id is not None else "dataset"
        raise ValueError(f"Too few labeled or unlabeled samples in {stratum_part}.")


def _validate_binary_or_nan(array: NDArray, name: str) -> None:
    array_float = array.astype(float)
    if not (np.isnan(array_float) | np.isin(array_float, [0.0, 1.0])).all():
        raise ValueError(f"'{name}' must only contain 0, 1, and np.nan values.")


def _validate_min_samples(values: NDArray, name: str, stratum_id: Optional[Hashable] = None) -> None:
    if len(values) < 2:
        if stratum_id is not None:
            raise ValueError(
                f"'{name}' must have at least 2 non-NaN values per stratum; "
                f"got {len(values)} in stratum '{stratum_id}'."
            )
        raise ValueError(f"'{name}' must have at least 2 non-NaN values; got {len(values)}.")
