from unittest.mock import call, patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.samplers.cost_optimal_random as cost_optimal_random_module
from glide.samplers import CostOptimalRandomSampler


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


@pytest.fixture
def fitted_sampler_high_MSE(sampler) -> CostOptimalRandomSampler:
    y_true = np.array([0.0, 10.0])
    y_proxy = np.array([2.0, 7.0])
    sampler.fit(y_true, y_proxy)
    return sampler


# --- fit ---


def test_fit_delegates_to_validation(sampler, y_true, y_proxy):
    with (
        patch.object(cost_optimal_random_module, "_validate_y_true_burn_in") as mock_validate_y_true_burn_in,
        patch.object(cost_optimal_random_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(cost_optimal_random_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
    ):
        sampler.fit(y_true, y_proxy)

        mock_validate_y_true_burn_in.assert_called_once()
        np.testing.assert_array_equal(mock_validate_y_true_burn_in.call_args[0][0], y_true)
        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], y_proxy)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y_true", "y_proxy"]}
        assert mock_validate_has_no_nan.call_count == 2
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args_list[0][0][0], y_proxy)
        assert mock_validate_has_no_nan.call_args_list[0][0][1] == "y_proxy"
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args_list[1][0][0], y_true)
        assert mock_validate_has_no_nan.call_args_list[1][0][1] == "y_true"


def test_fit_raises_on_zero_mse(sampler):
    with pytest.raises(ValueError, match="'y_proxy' predicts 'y_true' perfectly"):
        sampler.fit(y_true=np.array([1.0, 2.0]), y_proxy=np.array([1.0, 2.0]))


def test_fit_known_outputs(sampler, y_true, y_proxy):
    sampler.fit(y_true, y_proxy)

    expected_variance = 0.5
    expected_mse = 0.01

    assert sampler._y_true_variance == pytest.approx(expected_variance, abs=0.001)
    assert sampler._mean_squared_error == pytest.approx(expected_mse, abs=0.001)


# --- _compute_optimal_probability ---


def test_compute_optimal_probability_pi_one(fitted_sampler_high_MSE):
    pi = fitted_sampler_high_MSE._compute_optimal_probability(y_true_cost=1.0, y_proxy_cost=10.0)
    assert pi == 1.0
    assert fitted_sampler_high_MSE._y_true_variance == pytest.approx(50.0, abs=0.01)
    assert fitted_sampler_high_MSE._mean_squared_error == pytest.approx(6.5, abs=0.01)


def test_compute_optimal_probability_pi_known_value(fitted_sampler):
    pi = fitted_sampler._compute_optimal_probability(y_true_cost=10.0, y_proxy_cost=1.0)

    expected_pi = 0.0451

    assert pi == pytest.approx(expected_pi, abs=0.01)
    assert fitted_sampler._y_true_variance == pytest.approx(0.5, abs=0.01)
    assert fitted_sampler._mean_squared_error == pytest.approx(0.01, abs=0.01)


# --- sample ---


def test_sample_raises_if_fit_not_called(sampler):
    with pytest.raises(RuntimeError, match="Call fit\\(\\) before sample"):
        sampler.sample(n_samples=2, y_true_cost=10.0, y_proxy_cost=1.0, budget=5, random_seed=42)


def test_sample_delegates_to_validation(fitted_sampler):
    with (
        patch.object(cost_optimal_random_module, "_validate_is_integer") as mock_validate_is_integer,
        patch.object(cost_optimal_random_module, "_validate_strictly_positive") as mock_validate_strictly_positive,
        patch.object(cost_optimal_random_module, "_validate_bounds") as mock_validate_bounds,
    ):
        fitted_sampler.sample(n_samples=2, y_true_cost=10.0, y_proxy_cost=1.0, budget=10, random_seed=42)

        mock_validate_is_integer.assert_called_once_with(2, "n_samples")
        mock_validate_strictly_positive.assert_has_calls(
            [
                call(2, "n_samples"),
                call(10.0, "y_true_cost"),
                call(1.0, "y_proxy_cost"),
            ]
        )
        mock_validate_bounds.assert_called_once_with(
            10,
            "budget",
            lower=11.0,
            error_message="'budget' should be at least 11.0; got 10.",
        )


def test_sample_budget_too_small_raises(fitted_sampler):
    with pytest.raises(ValueError, match="'budget' should be at least"):
        fitted_sampler.sample(n_samples=2, y_true_cost=100.0, y_proxy_cost=1.0, budget=1, random_seed=42)


def test_sample_known_output(fitted_sampler):
    n_samples = 2
    pi, xi = fitted_sampler.sample(n_samples=n_samples, y_true_cost=10.0, y_proxy_cost=1.0, budget=15, random_seed=42)

    expected_pi = np.array([0.045, 0.045])
    expected_xi = np.array([0.0, 0.0])

    np.testing.assert_allclose(pi, expected_pi, atol=0.01)
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_known_output_truncated_samples(fitted_sampler):
    n_samples = 5
    pi, xi = fitted_sampler.sample(n_samples=n_samples, y_true_cost=10.0, y_proxy_cost=1.0, budget=11, random_seed=6)

    expected_pi = np.array([0.0, 0.0, 0.045, 0.0, 0.0])
    expected_xi = np.array([np.nan, np.nan, 1.0, np.nan, np.nan])

    np.testing.assert_allclose(pi, expected_pi, atol=0.01, equal_nan=True)
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_reproducibility(fitted_sampler):
    pi1, xi1 = fitted_sampler.sample(n_samples=2, y_true_cost=10.0, y_proxy_cost=1.0, budget=15, random_seed=42)
    pi2, xi2 = fitted_sampler.sample(n_samples=2, y_true_cost=10.0, y_proxy_cost=1.0, budget=15, random_seed=42)

    np.testing.assert_array_equal(pi1, pi2)
    np.testing.assert_array_equal(xi1, xi2)
