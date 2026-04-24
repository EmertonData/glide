import numpy as np
import pytest
from numpy.typing import NDArray

from glide.samplers.cost_optimal_random import CostOptimalRandomSampler


@pytest.fixture
def y_true() -> NDArray:
    return np.array([1.0, 2.0])


@pytest.fixture
def y_proxy() -> NDArray:
    return np.array([1.1, 1.9])


@pytest.fixture
def sampler() -> CostOptimalRandomSampler:
    return CostOptimalRandomSampler()


@pytest.fixture
def fitted_sampler(sampler, y_true, y_proxy) -> CostOptimalRandomSampler:
    sampler.fit(y_true, y_proxy)
    return sampler


# --- fit ---


def test_fit_valid_fit_returns_self(sampler, y_true, y_proxy):
    result = sampler.fit(y_true, y_proxy)
    assert result is sampler


def test_fit_raises_on_empty_y_true(sampler):
    with pytest.raises(ValueError, match=r"`y_true` must not be empty"):
        sampler.fit(np.array([]), np.array([1.0]))


def test_fit_raises_on_length_mismatch(sampler):
    with pytest.raises(ValueError, match="must have the same length"):
        sampler.fit(np.array([1.0, 2.0]), np.array([1.1]))


def test_fit_raises_on_nan_in_input(sampler):
    with pytest.raises(ValueError, match="Input values contain NaN"):
        sampler.fit(np.array([1.0, np.nan]), np.array([1.1, 1.9]))

    with pytest.raises(ValueError, match="Input values contain NaN"):
        sampler.fit(np.array([1.0, 2.0]), np.array([1.1, np.nan]))


def test_fit_raises_on_zero_variance_y_true(sampler):
    with pytest.raises(ValueError, match="Input ground-truth values have zero variance"):
        sampler.fit(np.array([1.0, 1.0]), np.array([1.1, 1.1]))


def test_fit_raises_on_zero_mse(sampler):
    with pytest.raises(ValueError, match="Proxy and ground-truth values match perfectly"):
        sampler.fit(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


# --- _compute_optimal_probability ---


def test_compute_optimal_probability_pi_equals_one_when_mse_exceeds_threshold(sampler):
    y_true = np.array([0.0, 10.0])
    y_proxy = np.array([2.0, 7.0])
    sampler.fit(y_true, y_proxy)

    y_true_cost = 1.0
    y_proxy_cost = 10.0
    pi = sampler._compute_optimal_probability(y_true_cost, y_proxy_cost)

    assert pi == 1.0


def test_compute_optimal_probability_pi_formula_when_mse_below_threshold(sampler):
    y_true = np.array([0.0, 1.0, 2.0])
    y_proxy = np.array([0.1, 1.1, 1.9])
    sampler.fit(y_true, y_proxy)

    y_true_cost = 10.0
    y_proxy_cost = 1.0
    pi = sampler._compute_optimal_probability(y_true_cost, y_proxy_cost)

    assert sampler._y_true_variance == pytest.approx(1.0, abs=0.01)
    assert sampler._mean_squared_error == pytest.approx(0.01, abs=0.01)
    assert pi == pytest.approx(0.0318, abs=0.01)


# --- sample ---


def test_sample_raises_if_fit_not_called(sampler, y_proxy):
    with pytest.raises(RuntimeError, match="Call fit\\(\\) before sample"):
        sampler.sample(y_proxy=y_proxy, y_true_cost=10.0, y_proxy_cost=1.0, budget=5, random_seed=42)


@pytest.mark.parametrize("y_true_cost", [0.0, -1.0])
def test_sample_invalid_y_true_cost(fitted_sampler, y_proxy, y_true_cost):
    with pytest.raises(ValueError, match=r"`y_true_cost` must be strictly positive"):
        fitted_sampler.sample(y_proxy=y_proxy, y_true_cost=y_true_cost, y_proxy_cost=1.0, budget=5, random_seed=42)


@pytest.mark.parametrize("y_proxy_cost", [0.0, -1.0])
def test_sample_invalid_y_proxy_cost(fitted_sampler, y_proxy, y_proxy_cost):
    with pytest.raises(ValueError, match=r"`y_proxy_cost` must be strictly positive"):
        fitted_sampler.sample(y_proxy=y_proxy, y_true_cost=10.0, y_proxy_cost=y_proxy_cost, budget=5, random_seed=42)


@pytest.mark.parametrize("budget", [0.0, -1.0])
def test_sample_invalid_budget(fitted_sampler, y_proxy, budget):
    with pytest.raises(ValueError, match=r"`budget` must be strictly positive"):
        fitted_sampler.sample(y_proxy=y_proxy, y_true_cost=10.0, y_proxy_cost=1.0, budget=budget, random_seed=42)


def test_sample_budget_too_small_raises(fitted_sampler, y_proxy):
    with pytest.raises(ValueError, match="Budget .* is too small"):
        fitted_sampler.sample(y_proxy=y_proxy, y_true_cost=100.0, y_proxy_cost=1.0, budget=1, random_seed=42)


def test_sample_returns_tuple_with_correct_shapes(fitted_sampler, y_proxy):
    indices, xi, pi = fitted_sampler.sample(
        y_proxy=y_proxy, y_true_cost=10.0, y_proxy_cost=1.0, budget=5, random_seed=42
    )

    assert isinstance(indices, np.ndarray)
    assert isinstance(xi, np.ndarray)
    assert isinstance(pi, float)
    assert len(indices) <= len(y_proxy)
    assert len(xi) == len(indices)
    assert 0.0 < pi <= 1.0
    assert np.all(indices[:-1] <= indices[1:])
    assert np.isin(xi, [0.0, 1.0]).all()


def test_sample_is_reproducible(fitted_sampler, y_proxy):
    indices1, xi1, pi1 = fitted_sampler.sample(
        y_proxy=y_proxy, y_true_cost=10.0, y_proxy_cost=1.0, budget=5, random_seed=42
    )
    indices2, xi2, pi2 = fitted_sampler.sample(
        y_proxy=y_proxy, y_true_cost=10.0, y_proxy_cost=1.0, budget=5, random_seed=42
    )

    np.testing.assert_array_equal(indices1, indices2)
    np.testing.assert_array_equal(xi1, xi2)
    assert pi1 == pi2
