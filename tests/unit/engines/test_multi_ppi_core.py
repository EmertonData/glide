from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from glide.engines.multi_ppi_core import (
    _compute_mean_estimate,
    _compute_std_estimate,
    _compute_tuning_parameter,
)


@pytest.fixture
def y_true() -> NDArray:
    return np.array([2.0, 4.0])


@pytest.fixture
def y_proxies_labeled() -> NDArray:
    return np.array([[1.0, 2.0], [3.0, 2.0]])


@pytest.fixture
def y_proxies_unlabeled() -> NDArray:
    return np.array([[1.5, 3.0], [3.5, 3.0]])


# --- _compute_tuning_parameter ---


def test_compute_tuning_parameter_delegates_to_validation(y_true, y_proxies_labeled, y_proxies_unlabeled):
    with patch("glide.engines.multi_ppi_core._validate_non_constant") as mock_validate_non_constant:
        _compute_tuning_parameter(y_true, y_proxies_labeled, y_proxies_unlabeled, power_tuning=True)
    expected_y_proxies_all = np.vstack([y_proxies_labeled, y_proxies_unlabeled])
    assert mock_validate_non_constant.call_count == 2
    np.testing.assert_array_equal(mock_validate_non_constant.call_args_list[0][0][0], expected_y_proxies_all[:, 0])
    assert mock_validate_non_constant.call_args_list[0][0][1] == "'y_proxies' column 0 values are constant."
    np.testing.assert_array_equal(mock_validate_non_constant.call_args_list[1][0][0], expected_y_proxies_all[:, 1])
    assert mock_validate_non_constant.call_args_list[1][0][1] == "'y_proxies' column 1 values are constant."


def test_compute_tuning_parameter_power_tuning_false(y_true, y_proxies_labeled, y_proxies_unlabeled):
    expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    result = _compute_tuning_parameter(y_true, y_proxies_labeled, y_proxies_unlabeled, power_tuning=False)
    np.testing.assert_allclose(result, expected)


def test_compute_tuning_parameter_known_value(y_true, y_proxies_labeled, y_proxies_unlabeled):
    expected = np.array([0.75, -0.375])
    result = _compute_tuning_parameter(y_true, y_proxies_labeled, y_proxies_unlabeled, power_tuning=True)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_compute_tuning_parameter_singular_matrix_raises(y_true, y_proxies_labeled, y_proxies_unlabeled):
    identical_labeled = np.column_stack([y_proxies_labeled[:, 0], y_proxies_labeled[:, 0]])
    identical_unlabeled = np.column_stack([y_proxies_unlabeled[:, 0], y_proxies_unlabeled[:, 0]])
    with pytest.raises(ValueError, match="singular"):
        _compute_tuning_parameter(y_true, identical_labeled, identical_unlabeled, power_tuning=True)


# --- _compute_mean_estimate ---


def test_compute_mean_estimate_known_values(y_true, y_proxies_labeled, y_proxies_unlabeled):
    lambdas_ = np.array([0.5, 0.5])
    expected = 3.75
    result = _compute_mean_estimate(y_true, y_proxies_labeled, y_proxies_unlabeled, lambdas_)
    assert result == pytest.approx(expected)


# --- _compute_std_estimate ---


def test_compute_std_estimate_known_values(y_true, y_proxies_labeled, y_proxies_unlabeled):
    lambdas_ = np.array([0.5, 0.5])
    expected = np.sqrt(0.5)
    result = _compute_std_estimate(y_true, y_proxies_labeled, y_proxies_unlabeled, lambdas_)
    assert result == pytest.approx(expected)
