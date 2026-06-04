from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.samplers.cost_optimal as cost_optimal_module
from glide.samplers import CostOptimalSampler


@pytest.fixture
def y_true() -> NDArray:
    return np.array([1.0, 2.0])


@pytest.fixture
def uncertainties() -> NDArray:
    return np.array([0.1, 0.4])


@pytest.fixture
def sampler() -> CostOptimalSampler:
    return CostOptimalSampler()


@pytest.fixture
def fitted_sampler(sampler, y_true) -> CostOptimalSampler:
    sampler.fit(y_true)
    return sampler


@pytest.fixture
def fitted_sampler_small_variance(sampler) -> CostOptimalSampler:
    sampler.fit(np.array([0.0, 0.001]))
    return sampler


# --- fit ---


def test_fit_delegates_to_validation(sampler, y_true):
    with (
        patch.object(cost_optimal_module, "_validate_y_true_burn_in") as mock_validate_y_true_burn_in,
    ):
        sampler.fit(y_true)
        mock_validate_y_true_burn_in.assert_called_once_with(y_true)


def test_fit_known_variance(sampler):
    sampler.fit(np.array([1.0, 2.0, 3.0]))
    expected_variance = 1.0
    assert sampler._y_true_variance == pytest.approx(expected_variance, abs=0.001)


# --- _compute_gamma ---


def test_compute_gamma_known_output(fitted_sampler, uncertainties):
    tau = 0.4
    gamma = fitted_sampler._compute_gamma(tau, uncertainties, y_true_cost=10.0, y_proxy_cost=1.0)
    expected_gamma = 0.4909
    assert gamma == pytest.approx(expected_gamma, abs=0.001)


def test_compute_gamma_known_output_small_variance(fitted_sampler_small_variance, uncertainties):
    tau = 0.4
    gamma = fitted_sampler_small_variance._compute_gamma(tau, uncertainties, y_true_cost=10.0, y_proxy_cost=1.0)
    expected_gamma = 2.5
    assert gamma == pytest.approx(expected_gamma, abs=0.001)


# --- _compute_per_sample_probabilities ---


def test_compute_per_sample_probabilities_known_output(fitted_sampler, uncertainties):
    tau = 0.4
    gamma = 0.4909
    probs = fitted_sampler._compute_per_sample_probabilities(tau, gamma, uncertainties)

    expected_probs = np.array([0.049, 0.197])
    np.testing.assert_allclose(probs, expected_probs, atol=0.001)


# --- _compute_objective ---


def test_compute_objective_known_output(fitted_sampler, uncertainties):
    tau = 0.4
    objective = fitted_sampler._compute_objective(tau, uncertainties, y_true_cost=10.0, y_proxy_cost=1.0)
    expected_value = 2.058
    assert objective == pytest.approx(expected_value, abs=0.001)


# --- _find_optimal_threshold ---


def test_find_optimal_threshold_known_output(fitted_sampler, uncertainties):
    tau_star = fitted_sampler._find_optimal_threshold(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0)
    expected_tau_star = 0.4
    assert tau_star == pytest.approx(expected_tau_star, abs=0.001)


def test_find_optimal_threshold_zero_y_proxy_cost_known_output(fitted_sampler, uncertainties):
    tau_star = fitted_sampler._find_optimal_threshold(uncertainties, y_true_cost=10.0, y_proxy_cost=0.0)
    expected_tau_star = 0.1
    assert tau_star == pytest.approx(expected_tau_star, abs=0.001)


# --- sample ---


def test_sample_raises_if_fit_not_called(sampler, uncertainties):
    with pytest.raises(RuntimeError, match="Call fit\\(\\) before sample\\(\\)"):
        sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0, budget=5, random_seed=42)


def test_sample_delegates_to_validation(fitted_sampler, uncertainties):
    with (
        patch.object(cost_optimal_module, "_validate_strictly_positive") as mock_validate_strictly_positive,
        patch.object(cost_optimal_module, "_validate_uncertainties") as mock_validate_uncertainties,
        patch.object(cost_optimal_module, "_validate_non_constant") as mock_validate_non_constant,
        patch.object(cost_optimal_module, "_validate_bounds") as mock_validate_bounds,
    ):
        fitted_sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=0.0, budget=10, random_seed=42)

        mock_validate_strictly_positive.assert_called_once_with(10.0, "y_true_cost")

        mock_validate_uncertainties.assert_called_once()
        np.testing.assert_array_equal(mock_validate_uncertainties.call_args[0][0], uncertainties)
        mock_validate_non_constant.assert_called_once()
        np.testing.assert_array_equal(mock_validate_non_constant.call_args[0][0], uncertainties)
        expected_msg = (
            "All uncertainty values are equal and 'y_proxy_cost' is zero."
            " Provide non-constant uncertainties or set 'y_proxy_cost' to a positive value."
        )
        assert mock_validate_non_constant.call_args[0][1] == expected_msg
        mock_validate_bounds.assert_called_once_with(
            10,
            "budget",
            lower=10.0,
            error_message="'budget' should be at least 10.0; got 10.",
        )


def test_sample_negative_y_proxy_cost(fitted_sampler, uncertainties):
    with pytest.raises(ValueError, match="'y_proxy_cost' must be non-negative"):
        fitted_sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=-1, budget=5, random_seed=42)


def test_sample_budget_too_small_raises(fitted_sampler, uncertainties):
    with pytest.raises(ValueError, match="'budget' should be at least"):
        fitted_sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0, budget=1, random_seed=42)


def test_sample_known_output(fitted_sampler, uncertainties):
    pi, xi = fitted_sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0, budget=20, random_seed=42)

    expected_pi = np.array([0.049, 0.196])
    expected_xi = np.array([0.0, 0.0])

    np.testing.assert_allclose(pi, expected_pi, atol=0.001)
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_known_output_truncated_samples(fitted_sampler, uncertainties):
    pi, xi = fitted_sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0, budget=11, random_seed=2)

    expected_pi = np.array([0.049, 0.0])

    np.testing.assert_allclose(pi, expected_pi, atol=0.001)
    assert xi[0] == 0.0
    assert np.isnan(xi[1])


def test_sample_reproducibility(fitted_sampler, uncertainties):
    pi1, xi1 = fitted_sampler.sample(uncertainties, y_true_cost=1.0, y_proxy_cost=0.01, budget=1.5, random_seed=42)
    pi2, xi2 = fitted_sampler.sample(uncertainties, y_true_cost=1.0, y_proxy_cost=0.01, budget=1.5, random_seed=42)

    np.testing.assert_array_equal(pi1, pi2)
    np.testing.assert_array_equal(xi1, xi2)


def test_sample_different_seeds_results_differ(fitted_sampler, uncertainties):
    pi1, xi1 = fitted_sampler.sample(uncertainties, y_true_cost=1.0, y_proxy_cost=0.9, budget=20, random_seed=0)
    pi2, xi2 = fitted_sampler.sample(uncertainties, y_true_cost=1.0, y_proxy_cost=0.9, budget=20, random_seed=1)

    assert (not np.array_equal(pi1, pi2)) or (not np.array_equal(xi1, xi2))
