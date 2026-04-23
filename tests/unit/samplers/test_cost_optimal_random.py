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


# --- fit ---


def test_fit_valid_fit_returns_self(sampler, y_true, y_proxy):
    result = sampler.fit(y_true, y_proxy)
    assert result is sampler


def test_fit_raises_on_empty_y_true(sampler):
    with pytest.raises(ValueError, match="'y_true' must not be empty"):
        sampler.fit(np.array([]), np.array([1.0]))


def test_fit_raises_on_empty_y_proxy(sampler):
    with pytest.raises(ValueError, match="'y_proxy' must not be empty"):
        sampler.fit(np.array([1.0]), np.array([]))


def test_fit_raises_on_length_mismatch(sampler):
    with pytest.raises(ValueError, match="must have the same length"):
        sampler.fit(np.array([1.0, 2.0]), np.array([1.1]))


def test_fit_raises_on_nan_in_y_true(sampler):
    with pytest.raises(ValueError, match="'y_true' must not contain NaN"):
        sampler.fit(np.array([1.0, np.nan]), np.array([1.1, 1.9]))


def test_fit_raises_on_nan_in_y_proxy(sampler):
    with pytest.raises(ValueError, match="'y_proxy' must not contain NaN"):
        sampler.fit(np.array([1.0, 2.0]), np.array([1.1, np.nan]))


def test_fit_raises_on_zero_variance_y_true(sampler):
    with pytest.raises(ValueError, match="Var\\(H\\) is zero"):
        sampler.fit(np.array([1.0, 1.0]), np.array([1.1, 1.1]))


def test_fit_raises_on_zero_mse(sampler):
    with pytest.raises(ValueError, match="MSE\\(H, G\\) is zero"):
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

    y_true_variance = sampler._y_true_variance
    mean_squared_error = sampler._mean_squared_error
    threshold = y_true_cost / (y_true_cost + y_proxy_cost) * y_true_variance

    assert mean_squared_error < threshold
    expected_pi = np.sqrt((y_proxy_cost / y_true_cost) * mean_squared_error / (y_true_variance - mean_squared_error))
    assert pi == pytest.approx(expected_pi, abs=1e-10)


# --- sample ---


def test_sample_raises_if_fit_not_called(sampler, y_proxy):
    with pytest.raises(ValueError, match="Call fit\\(\\) before sample"):
        sampler.sample(y_proxy, 10.0, 1.0, 5, 42)


def test_sample_invalid_y_true_cost_zero(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    with pytest.raises(ValueError, match="'y_true_cost' must be strictly positive"):
        sampler.sample(y_proxy, 0.0, 1.0, 5, 42)


def test_sample_invalid_y_true_cost_negative(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    with pytest.raises(ValueError, match="'y_true_cost' must be strictly positive"):
        sampler.sample(y_proxy, -1.0, 1.0, 5, 42)


def test_sample_invalid_y_proxy_cost_zero(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    with pytest.raises(ValueError, match="'y_proxy_cost' must be strictly positive"):
        sampler.sample(y_proxy, 10.0, 0.0, 5, 42)


def test_sample_invalid_y_proxy_cost_negative(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    with pytest.raises(ValueError, match="'y_proxy_cost' must be strictly positive"):
        sampler.sample(y_proxy, 10.0, -1.0, 5, 42)


def test_sample_invalid_budget_zero(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    with pytest.raises(ValueError, match="'budget' must be a strictly positive integer"):
        sampler.sample(y_proxy, 10.0, 1.0, 0, 42)


def test_sample_invalid_budget_negative(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    with pytest.raises(ValueError, match="'budget' must be a strictly positive integer"):
        sampler.sample(y_proxy, 10.0, 1.0, -1, 42)


def test_sample_invalid_budget_float(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    with pytest.raises(ValueError, match="'budget' must be a strictly positive integer"):
        sampler.sample(y_proxy, 10.0, 1.0, 1.5, 42)


def test_sample_invalid_budget_boolean(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    with pytest.raises(ValueError, match="'budget' must be a strictly positive integer"):
        sampler.sample(y_proxy, 10.0, 1.0, True, 42)


def test_sample_budget_too_small_raises(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    with pytest.raises(ValueError, match="Budget .* is too small"):
        sampler.sample(y_proxy, 100.0, 1.0, 1, 42)


def test_sample_returns_tuple_with_correct_shapes(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    indices, xi, pi = sampler.sample(y_proxy, 10.0, 1.0, 5, 42)

    assert isinstance(indices, np.ndarray)
    assert isinstance(xi, np.ndarray)
    assert isinstance(pi, float)
    assert len(indices) <= len(y_proxy)
    assert len(xi) == len(indices)
    assert 0.0 < pi <= 1.0


def test_sample_indices_are_sorted(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    indices, _, _ = sampler.sample(y_proxy, 10.0, 1.0, 5, 42)

    assert np.all(indices[:-1] <= indices[1:])


def test_sample_indicators_are_zero_or_one(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    _, xi, _ = sampler.sample(y_proxy, 10.0, 1.0, 5, 42)

    assert np.isin(xi, [0.0, 1.0]).all()


def test_sample_returns_all_indices_when_T_exceeds_N(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    indices, _, _ = sampler.sample(y_proxy, 1000.0, 0.1, 10, 42)

    assert len(indices) == len(y_proxy)
    np.testing.assert_array_equal(indices, np.arange(len(y_proxy)))


def test_sample_returns_subset_indices_when_T_less_than_N(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    indices, _, _ = sampler.sample(y_proxy, 10.0, 1.0, 2, 42)

    assert len(indices) < len(y_proxy)


def test_sample_is_reproducible(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    indices1, xi1, pi1 = sampler.sample(y_proxy, 10.0, 1.0, 5, 42)
    indices2, xi2, pi2 = sampler.sample(y_proxy, 10.0, 1.0, 5, 42)

    np.testing.assert_array_equal(indices1, indices2)
    np.testing.assert_array_equal(xi1, xi2)
    assert pi1 == pi2
