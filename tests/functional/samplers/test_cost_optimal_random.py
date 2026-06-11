import numpy as np
import pytest

from glide.samplers import CostOptimalRandomSampler


@pytest.fixture
def sampler() -> CostOptimalRandomSampler:
    y_true = np.array([1.0, 2.0])
    y_proxy = np.array([1.1, 1.9])
    return CostOptimalRandomSampler().fit(y_true, y_proxy)


def test_total_cost_never_exceeds_max_cost(sampler):
    n_samples = 50
    y_true_cost = 10.0
    y_proxy_cost = 1.0
    max_cost = 150
    n_trials = 500

    for random_seed in range(n_trials):
        _, xi = sampler.sample(
            n_samples=n_samples,
            y_true_cost=y_true_cost,
            y_proxy_cost=y_proxy_cost,
            max_cost=max_cost,
            random_seed=random_seed,
        )
        included = ~np.isnan(xi)
        actual_cost = np.sum(xi[included]) * y_true_cost + np.sum(included) * y_proxy_cost
        assert actual_cost <= max_cost
