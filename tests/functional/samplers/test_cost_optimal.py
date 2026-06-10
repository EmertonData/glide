import numpy as np
import pytest

from glide.samplers import CostOptimalRandomSampler, CostOptimalSampler


@pytest.fixture
def sampler() -> CostOptimalSampler:
    return CostOptimalSampler().fit(np.array([1.0, 2.0]))


def test_total_cost_never_exceeds_budget(sampler):
    n_samples = 50
    y_true_cost = 10.0
    y_proxy_cost = 1.0
    total_cost = 150
    n_trials = 500

    uncertainties = np.linspace(0.1, 1.0, n_samples)

    for random_seed in range(n_trials):
        _, xi = sampler.sample(
            uncertainties,
            y_true_cost=y_true_cost,
            y_proxy_cost=y_proxy_cost,
            total_cost=total_cost,
            random_seed=random_seed,
        )
        included = ~np.isnan(xi)
        actual_cost = np.sum(xi[included]) * y_true_cost + np.sum(included) * y_proxy_cost
        assert actual_cost <= total_cost


def test_cost_optimal_matches_random_sampler_for_uniform_uncertainties():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_proxy = y_true + 0.1  # constant offset of 0.1 -> uniform RMSE = 0.1
    n_samples = 100
    uniform_uncertainties = np.full(n_samples, 0.1)
    y_true_cost = 10.0
    y_proxy_cost = 1.0
    total_cost = 500

    random_seed = 0

    random_sampler = CostOptimalRandomSampler().fit(y_true, y_proxy)
    pi_random, xi_random = random_sampler.sample(
        n_samples=n_samples,
        y_true_cost=y_true_cost,
        y_proxy_cost=y_proxy_cost,
        total_cost=total_cost,
        random_seed=random_seed,
    )

    cost_optimal_sampler = CostOptimalSampler().fit(y_true)
    pi_cost_optimal, xi_cost_optimal = cost_optimal_sampler.sample(
        uniform_uncertainties,
        y_true_cost=y_true_cost,
        y_proxy_cost=y_proxy_cost,
        total_cost=total_cost,
        random_seed=random_seed,
    )

    np.testing.assert_allclose(pi_cost_optimal, pi_random, atol=0.001)
    np.testing.assert_array_equal(xi_cost_optimal, xi_random)
