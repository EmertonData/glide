import numpy as np
import pytest
from numpy.typing import NDArray

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


def test_fit_raises_on_empty_y_true(sampler):
    with pytest.raises(ValueError, match="non-empty"):
        sampler.fit(np.array([]))


def test_fit_raises_on_nan_in_y_true(sampler):
    with pytest.raises(ValueError, match="NaN"):
        sampler.fit(np.array([1.0, np.nan]))


def test_fit_raises_on_zero_variance_y_true(sampler):
    with pytest.raises(ValueError, match="zero variance"):
        sampler.fit(np.array([1.0, 1.0]))


def test_fit_known_variance(sampler):
    sampler.fit(np.array([1.0, 2.0, 3.0]))
    expected_variance = 1.0
    assert sampler._y_true_variance == pytest.approx(expected_variance, abs=0.001)


# --- _compute_gamma ---


def test_compute_gamma_known_output(fitted_sampler, uncertainties):
    tau = np.sqrt(0.4)
    gamma = fitted_sampler._compute_gamma(tau, uncertainties, y_true_cost=10.0, y_proxy_cost=1.0)
    expected_gamma = 0.6325
    assert gamma == pytest.approx(expected_gamma, abs=0.001)


def test_compute_gamma_known_output_small_variance(fitted_sampler_small_variance, uncertainties):
    tau = np.sqrt(0.4)
    gamma = fitted_sampler_small_variance._compute_gamma(tau, uncertainties, y_true_cost=10.0, y_proxy_cost=1.0)
    expected_gamma = 1.5811
    assert gamma == pytest.approx(expected_gamma, abs=0.001)


# --- _compute_per_sample_probabilities ---


def test_compute_per_sample_probabilities_known_output(fitted_sampler, uncertainties):
    tau = np.sqrt(0.4)
    gamma = np.sqrt(0.4)
    probs = fitted_sampler._compute_per_sample_probabilities(tau, gamma, uncertainties)

    expected_probs = np.array([0.2, 0.4])
    np.testing.assert_allclose(probs, expected_probs, atol=0.001)


# --- _compute_objective ---


def test_compute_objective_known_output(fitted_sampler, uncertainties):
    tau = np.sqrt(0.4)
    objective = fitted_sampler._compute_objective(tau, uncertainties, y_true_cost=10.0, y_proxy_cost=1.0)
    expected_value = 4.0
    assert objective == pytest.approx(expected_value, abs=0.001)


# --- _find_optimal_threshold ---


def test_find_optimal_threshold_known_output(fitted_sampler, uncertainties):
    tau_star = fitted_sampler._find_optimal_threshold(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0)
    expected_tau_star = 0.6325
    assert tau_star == pytest.approx(expected_tau_star, abs=0.001)


# --- sample ---


def test_sample_raises_if_fit_not_called(sampler, uncertainties):
    with pytest.raises(RuntimeError, match="Call fit\\(\\) before sample\\(\\)"):
        sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0, budget=5, random_seed=42)


@pytest.mark.parametrize("cost", [0.0, -1.0])
def test_sample_invalid_y_true_cost(fitted_sampler, uncertainties, cost):
    with pytest.raises(ValueError, match="'y_true_cost' must be strictly positive"):
        fitted_sampler.sample(uncertainties, y_true_cost=cost, y_proxy_cost=1.0, budget=5, random_seed=42)


def test_sample_negative_y_proxy_cost(fitted_sampler, uncertainties):
    with pytest.raises(ValueError, match="'y_proxy_cost' must be non-negative"):
        fitted_sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=-1, budget=5, random_seed=42)


@pytest.mark.parametrize("budget", [0, -1])
def test_sample_invalid_budget(fitted_sampler, uncertainties, budget):
    with pytest.raises(ValueError, match="'budget' must be strictly positive"):
        fitted_sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0, budget=budget, random_seed=42)


def test_sample_budget_too_small_raises(fitted_sampler, uncertainties):
    with pytest.raises(ValueError, match="Budget .* is too small"):
        fitted_sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0, budget=1, random_seed=42)


def test_sample_raises_on_nan_uncertainties(fitted_sampler):
    with pytest.raises(ValueError, match="NaN"):
        fitted_sampler.sample(np.array([0.1, np.nan]), y_true_cost=10.0, y_proxy_cost=1.0, budget=5, random_seed=42)


def test_sample_raises_on_non_positive_uncertainties(fitted_sampler):
    with pytest.raises(ValueError, match="non-positive value"):
        fitted_sampler.sample(np.array([0.1, 0.0]), y_true_cost=10.0, y_proxy_cost=1.0, budget=5, random_seed=42)


def test_sample_known_output(fitted_sampler, uncertainties):
    pi, xi = fitted_sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0, budget=10, random_seed=42)

    expected_pi = np.array([0.2, 0.4])
    expected_xi = np.array([0.0, 0.0])

    np.testing.assert_allclose(pi, expected_pi, atol=0.01)
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_known_output_truncated_samples(fitted_sampler, uncertainties):
    pi, xi = fitted_sampler.sample(uncertainties, y_true_cost=10.0, y_proxy_cost=1.0, budget=5, random_seed=42)

    expected_pi = np.array([0.2, 0.0])

    np.testing.assert_allclose(pi, expected_pi, atol=0.01)
    assert xi[0] == 0.0
    assert np.isnan(xi[1])


def test_sample_reproducibility(fitted_sampler, uncertainties):
    pi1, xi1 = fitted_sampler.sample(uncertainties, y_true_cost=1.0, y_proxy_cost=0.01, budget=1.5, random_seed=42)
    pi2, xi2 = fitted_sampler.sample(uncertainties, y_true_cost=1.0, y_proxy_cost=0.01, budget=1.5, random_seed=42)

    np.testing.assert_array_equal(pi1, pi2)
    np.testing.assert_array_equal(xi1, xi2)
