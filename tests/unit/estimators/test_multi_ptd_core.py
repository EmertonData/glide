import numpy as np
import pytest
from numpy.typing import NDArray

from glide.estimators.multi_ptd_core import (
    _compute_bootstrap_mean_estimates,
    _compute_tuning_parameters,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def bootstrap_y_true_means() -> NDArray:
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def bootstrap_y_proxies_labeled_means() -> NDArray:
    return np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 2.0]])


@pytest.fixture
def cov_matrix_proxies_unlabeled() -> NDArray:
    return np.array([[0.5, 0.1], [0.1, 0.3]])


# --- _compute_tuning_parameters ---


def test_compute_tuning_parameters_returns_uniform_when_power_tuning_false(
    bootstrap_y_true_means,
    bootstrap_y_proxies_labeled_means,
    cov_matrix_proxies_unlabeled,
):
    result = _compute_tuning_parameters(
        bootstrap_y_true_means, bootstrap_y_proxies_labeled_means, cov_matrix_proxies_unlabeled, power_tuning=False
    )
    expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    np.testing.assert_allclose(result, expected)


def test_compute_tuning_parameters_known_value(
    bootstrap_y_true_means,
    bootstrap_y_proxies_labeled_means,
    cov_matrix_proxies_unlabeled,
):
    result = _compute_tuning_parameters(
        bootstrap_y_true_means, bootstrap_y_proxies_labeled_means, cov_matrix_proxies_unlabeled, power_tuning=True
    )
    expected = np.array([0.441810, 0.010776])
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_compute_tuning_parameters_singular_matrix_raises(
    bootstrap_y_true_means,
):
    identical_proxies_labeled_means = np.array([[2.0, 2.0], [4.0, 4.0], [6.0, 6.0]])
    identical_cov_matrix_unlabeled = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="singular"):
        _compute_tuning_parameters(
            bootstrap_y_true_means, identical_proxies_labeled_means, identical_cov_matrix_unlabeled, power_tuning=True
        )


# --- _compute_bootstrap_mean_estimates ---


def test_compute_bootstrap_mean_estimates_known_result(
    bootstrap_y_true_means,
    bootstrap_y_proxies_labeled_means,
    cov_matrix_proxies_unlabeled,
    rng,
):
    mean_proxies_unlabeled = np.array([5.0, 4.0])
    lambdas_ = np.array([0.6, 0.4])
    result = _compute_bootstrap_mean_estimates(
        bootstrap_y_true_means,
        bootstrap_y_proxies_labeled_means,
        mean_proxies_unlabeled,
        cov_matrix_proxies_unlabeled,
        lambdas_,
        rng,
    )
    expected = np.array([4.066053, 2.930598, 3.536451])
    np.testing.assert_allclose(result, expected, rtol=1e-5)
