import numpy as np
import pytest

from glide.estimators.ptd_core import (
    _compute_bootstrap_labeled_means,
    _compute_ptd_bootstrap_estimates,
    _compute_ptd_tuning_parameter,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# --- _compute_bootstrap_labeled_means ---


def test_compute_bootstrap_labeled_means_returns_correct_shapes(rng):
    y_true = np.array([5.0, 6.0, 7.0])
    y_proxy_labeled = np.array([4.5, 5.5, 6.5])
    y_true_means, y_proxy_means = _compute_bootstrap_labeled_means(y_true, y_proxy_labeled, n_bootstrap=10, rng=rng)
    assert y_true_means.shape == (10,)
    assert y_proxy_means.shape == (10,)


def test_compute_bootstrap_labeled_means_with_pi(rng):
    y_true = np.array([5.0, 6.0, 7.0])
    y_proxy_labeled = np.array([4.5, 5.5, 6.5])
    pi = np.array([0.2, 0.5, 0.3])
    y_true_means, y_proxy_means = _compute_bootstrap_labeled_means(
        y_true, y_proxy_labeled, n_bootstrap=10, rng=rng, pi=pi
    )
    assert y_true_means.shape == (10,)
    assert y_proxy_means.shape == (10,)


# --- _compute_ptd_tuning_parameter ---


def test_compute_ptd_tuning_parameter_returns_one_when_power_tuning_false():
    bootstrap_y_true_means = np.array([1.0, 2.0])
    bootstrap_y_proxy_labeled_means = np.array([1.0, 2.0])
    result = _compute_ptd_tuning_parameter(
        bootstrap_y_true_means, bootstrap_y_proxy_labeled_means, var_proxy_unlabeled=1.0, power_tuning=False
    )
    assert result == 1.0


def test_compute_ptd_tuning_parameter_known_value():
    bootstrap_y_true_means = np.array([1.0, 2.0, 3.0])
    bootstrap_y_proxy_labeled_means = np.array([2.0, 4.0, 6.0])
    result = _compute_ptd_tuning_parameter(
        bootstrap_y_true_means, bootstrap_y_proxy_labeled_means, var_proxy_unlabeled=0.5, power_tuning=True
    )
    expected = 2.0 / 4.5
    assert result == pytest.approx(expected)


# --- _compute_ptd_bootstrap_estimates ---


def test_compute_ptd_bootstrap_estimates_returns_correct_shape(rng):
    bootstrap_y_true_means = np.array([1.0, 2.0, 3.0])
    bootstrap_y_proxy_labeled_means = np.array([2.0, 4.0, 6.0])
    result = _compute_ptd_bootstrap_estimates(
        bootstrap_y_true_means,
        bootstrap_y_proxy_labeled_means,
        mean_proxy_unlabeled=5.0,
        var_proxy_unlabeled=0.5,
        lambda_=1.0,
        rng=rng,
    )
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_compute_ptd_bootstrap_estimates_lambda_zero_returns_rectifier(rng):
    bootstrap_y_true_means = np.array([2.0, 4.0])
    bootstrap_y_proxy_labeled_means = np.array([1.0, 3.0])
    result = _compute_ptd_bootstrap_estimates(
        bootstrap_y_true_means,
        bootstrap_y_proxy_labeled_means,
        mean_proxy_unlabeled=5.0,
        var_proxy_unlabeled=0.5,
        lambda_=0.0,
        rng=rng,
    )
    expected = np.array([2.0, 4.0])
    np.testing.assert_allclose(result, expected)
