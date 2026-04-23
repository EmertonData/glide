import numpy as np
import pytest
from numpy.typing import NDArray

from glide.estimators.ptd_core import (
    _compute_ptd_bootstrap_labeled_means,
    _compute_ptd_bootstrap_mean_estimates,
    _compute_ptd_tuning_parameter,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def bootstrap_y_true_means() -> NDArray:
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def bootstrap_y_proxy_labeled_means() -> NDArray:
    return np.array([2.0, 4.0, 6.0])


# --- _compute_ptd_bootstrap_labeled_means ---


def test_compute_ptd_bootstrap_labeled_means_known_result(rng):
    y_true = np.array([5.0, 6.0, 7.0])
    y_proxy_labeled = np.array([4.5, 5.5, 6.5])
    y_true_means, y_proxy_means = _compute_ptd_bootstrap_labeled_means(y_true, y_proxy_labeled, n_bootstrap=4, rng=rng)
    expected_y_true_means = np.array([6.333333, 5.0, 5.0, 6.666667])
    expected_y_proxy_means = np.array([5.833333, 4.5, 4.5, 6.166667])
    np.testing.assert_allclose(y_proxy_means, expected_y_proxy_means, rtol=1e-5)
    np.testing.assert_allclose(y_true_means, expected_y_true_means, rtol=1e-5)


# --- _compute_ptd_tuning_parameter ---


def test_compute_ptd_tuning_parameter_returns_one_when_power_tuning_false(
    bootstrap_y_true_means,
    bootstrap_y_proxy_labeled_means,
):
    result = _compute_ptd_tuning_parameter(
        bootstrap_y_true_means, bootstrap_y_proxy_labeled_means, var_proxy_unlabeled=1.0, power_tuning=False
    )
    assert result == 1.0


def test_compute_ptd_tuning_parameter_known_value(
    bootstrap_y_true_means,
    bootstrap_y_proxy_labeled_means,
):
    result = _compute_ptd_tuning_parameter(
        bootstrap_y_true_means, bootstrap_y_proxy_labeled_means, var_proxy_unlabeled=0.5, power_tuning=True
    )
    expected = 2.0 / 4.5
    assert result == pytest.approx(expected, abs=0.01)


# --- _compute_ptd_bootstrap_mean_estimates ---


def test_compute_ptd_bootstrap_mean_estimates_known_result(
    bootstrap_y_true_means, bootstrap_y_proxy_labeled_means, rng
):
    result = _compute_ptd_bootstrap_mean_estimates(
        bootstrap_y_true_means,
        bootstrap_y_proxy_labeled_means,
        mean_proxy_unlabeled=5.0,
        var_proxy_unlabeled=0.5,
        lambda_=1.0,
        rng=rng,
    )
    expected = np.array([4.088905, 2.906588, 2.452847])
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_compute_ptd_bootstrap_mean_estimates_lambda_zero_returns_true_means(
    bootstrap_y_true_means, bootstrap_y_proxy_labeled_means, rng
):
    result = _compute_ptd_bootstrap_mean_estimates(
        bootstrap_y_true_means,
        bootstrap_y_proxy_labeled_means,
        mean_proxy_unlabeled=5.0,
        var_proxy_unlabeled=0.5,
        lambda_=0.0,
        rng=rng,
    )
    np.testing.assert_allclose(result, bootstrap_y_true_means)
