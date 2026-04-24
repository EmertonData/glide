import numpy as np
import pytest
from numpy.typing import NDArray

from glide.estimators.ppi_core import (
    _compute_mean_estimate,
    _compute_std_estimate,
    _compute_tuning_parameter,
)


@pytest.fixture
def y_true() -> NDArray:
    return np.array([5.0, 6.0, 7.0])


@pytest.fixture
def y_proxy_labeled() -> NDArray:
    return np.array([4.5, 5.5, 6.5])


@pytest.fixture
def y_proxy_unlabeled() -> NDArray:
    return np.array([6.0, 7.0, 8.0])


# --- _compute_tuning_parameter ---


def test_compute_tuning_parameter_returns_one_when_power_tuning_false(y_true, y_proxy_labeled, y_proxy_unlabeled):
    result = _compute_tuning_parameter(y_true, y_proxy_labeled, y_proxy_unlabeled, power_tuning=False)
    assert result == 1.0


def test_compute_tuning_parameter_known_value(y_true, y_proxy_labeled, y_proxy_unlabeled):
    expected = 0.34
    result = _compute_tuning_parameter(y_true, y_proxy_labeled, y_proxy_unlabeled, power_tuning=True)
    assert result == pytest.approx(expected, abs=0.01)


# --- _compute_mean_estimate ---


def test_compute_mean_estimate_known_values(y_true, y_proxy_labeled, y_proxy_unlabeled):
    expected = 6.75
    result = _compute_mean_estimate(y_true, y_proxy_labeled, y_proxy_unlabeled, lambda_=0.5)
    assert result == pytest.approx(expected)


# --- _compute_std_estimate ---


def test_compute_std_estimate_known_values(y_true, y_proxy_labeled, y_proxy_unlabeled):
    expected = 0.41
    result = _compute_std_estimate(y_true, y_proxy_labeled, y_proxy_unlabeled, lambda_=0.5)
    assert result == pytest.approx(expected, abs=1e-2)
