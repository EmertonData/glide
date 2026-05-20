import numpy as np

from glide.samplers import CostOptimalRandomSampler, CostOptimalSampler


def test_cost_optimal_matches_random_sampler_for_uniform_uncertainties():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_proxy = y_true + 0.1  # constant offset of 0.1 -> uniform MSE = 0.01
    n_samples = 100
    uniform_uncertainties = np.full(n_samples, 0.01)
    y_true_cost = 10.0
    y_proxy_cost = 1.0
    budget = 500

    random_seed = 0

    random_sampler = CostOptimalRandomSampler().fit(y_true, y_proxy)
    pi_random, xi_random = random_sampler.sample(
        n_samples=n_samples,
        y_true_cost=y_true_cost,
        y_proxy_cost=y_proxy_cost,
        budget=budget,
        random_seed=random_seed,
    )

    cost_optimal_sampler = CostOptimalSampler().fit(y_true)
    pi_cost_optimal, xi_cost_optimal = cost_optimal_sampler.sample(
        uniform_uncertainties,
        y_true_cost=y_true_cost,
        y_proxy_cost=y_proxy_cost,
        budget=budget,
        random_seed=random_seed,
    )

    np.testing.assert_allclose(pi_cost_optimal, pi_random, atol=0.001)
    np.testing.assert_array_equal(xi_cost_optimal, xi_random)
