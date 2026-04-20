import numpy as np
import pytest

from glide.estimators.ptd_core import (
    _compute_bootstrap_labeled_means,
    _compute_ptd_bootstrap_estimates,
    _compute_ptd_tuning_scalar,
    _compute_unlabeled_proxy_mean,
    _compute_unlabeled_proxy_var,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# --- _compute_unlabeled_proxy_mean ---


def test_compute_unlabeled_proxy_mean_known_value():
    y_proxy_unlabeled = np.array([3.0, 5.0])
    result = _compute_unlabeled_proxy_mean(y_proxy_unlabeled)
    assert result == pytest.approx(4.0)


# --- _compute_unlabeled_proxy_var ---


def test_compute_unlabeled_proxy_var_known_value():
    y_proxy_unlabeled = np.array([3.0, 5.0])
    result = _compute_unlabeled_proxy_var(y_proxy_unlabeled)
    assert result == pytest.approx(1.0)


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


# --- _compute_ptd_tuning_scalar ---


def test_compute_ptd_tuning_scalar_returns_one_when_power_tuning_false():
    bootstraps = (np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    result = _compute_ptd_tuning_scalar(bootstraps, var_proxy_unlabeled=1.0, power_tuning=False)
    assert result == 1.0


def test_compute_ptd_tuning_scalar_known_value():
    bootstraps = (np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    result = _compute_ptd_tuning_scalar(bootstraps, var_proxy_unlabeled=0.5, power_tuning=True)
    expected = 2.0 / 4.5
    assert result == pytest.approx(expected)


# --- _compute_ptd_bootstrap_estimates ---


def test_compute_ptd_bootstrap_estimates_returns_correct_shape(rng):
    bootstraps = (np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    result = _compute_ptd_bootstrap_estimates(
        bootstraps,
        mean_proxy_unlabeled=5.0,
        var_proxy_unlabeled=0.5,
        lambda_=1.0,
        rng=rng,
    )
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_compute_ptd_bootstrap_estimates_lambda_zero_returns_rectifier(rng):
    bootstraps = (np.array([2.0, 4.0]), np.array([1.0, 3.0]))
    result = _compute_ptd_bootstrap_estimates(
        bootstraps,
        mean_proxy_unlabeled=5.0,
        var_proxy_unlabeled=0.5,
        lambda_=0.0,
        rng=rng,
    )
    expected = np.array([2.0, 4.0])
    np.testing.assert_allclose(result, expected)
