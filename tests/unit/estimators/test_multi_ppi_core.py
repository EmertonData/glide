import numpy as np
import pytest
from numpy.typing import NDArray

from glide.estimators.multi_ppi_core import (
    _compute_mean_estimate,
    _compute_std_estimate,
    _compute_tuning_parameters,
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


# --- _compute_tuning_parameters ---


def test_compute_tuning_parameters_power_tuning_false(y_true, y_proxies_labeled, y_proxies_unlabeled):
    expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    result = _compute_tuning_parameters(y_true, y_proxies_labeled, y_proxies_unlabeled, power_tuning=False)
    np.testing.assert_allclose(result, expected)


def test_compute_tuning_parameters_known_value(y_true, y_proxies_labeled, y_proxies_unlabeled):
    expected = np.array([0.75, -0.375])
    result = _compute_tuning_parameters(y_true, y_proxies_labeled, y_proxies_unlabeled, power_tuning=True)
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_compute_tuning_parameters_singular_matrix_raises(y_true, y_proxies_labeled, y_proxies_unlabeled):
    identical_labeled = np.column_stack([y_proxies_labeled[:, 0], y_proxies_labeled[:, 0]])
    identical_unlabeled = np.column_stack([y_proxies_unlabeled[:, 0], y_proxies_unlabeled[:, 0]])
    with pytest.raises(ValueError, match="singular"):
        _compute_tuning_parameters(y_true, identical_labeled, identical_unlabeled, power_tuning=True)


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
    assert result == pytest.approx(expected, abs=1e-10)
