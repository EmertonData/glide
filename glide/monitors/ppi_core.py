from typing import Union, overload

from numpy.typing import NDArray

from glide.estimators.ppi_core import _compute_tuning_parameter


def _compute_clipped_tuning_parameter(
    y_true_labeled: NDArray,
    y_proxy_labeled: NDArray,
    y_proxy_unlabeled: NDArray,
    power_tuning: bool,
    max_tuning_parameter: float,
) -> float:
    tuning_parameter = _compute_tuning_parameter(y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, power_tuning)
    clipped_tuning_parameter = min(max(tuning_parameter, 0.0), max_tuning_parameter)
    return clipped_tuning_parameter


@overload
def _normalize_to_unit_interval(values: float, max_tuning_parameter: float) -> float: ...
@overload
def _normalize_to_unit_interval(values: NDArray, max_tuning_parameter: float) -> NDArray: ...
def _normalize_to_unit_interval(
    values: Union[float, NDArray],
    max_tuning_parameter: float,
) -> Union[float, NDArray]:
    normalization = 1.0 + 2.0 * max_tuning_parameter
    normalized = (values + max_tuning_parameter) / normalization
    return normalized


@overload
def _denormalize_from_unit_interval(values: float, max_tuning_parameter: float) -> float: ...
@overload
def _denormalize_from_unit_interval(values: NDArray, max_tuning_parameter: float) -> NDArray: ...
def _denormalize_from_unit_interval(
    values: Union[float, NDArray],
    max_tuning_parameter: float,
) -> Union[float, NDArray]:
    normalization = 1.0 + 2.0 * max_tuning_parameter
    original = values * normalization - max_tuning_parameter
    return original
