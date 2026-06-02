import numpy as np
import pytest

from glide.samplers import ActiveSampler


@pytest.fixture
def sampler() -> ActiveSampler:
    return ActiveSampler()


def test_sample_never_exceeds_budget(sampler):
    n_samples = 50
    budget = 10
    n_trials = 500

    uncertainties = np.ones(n_samples)

    for random_seed in range(n_trials):
        pi, xi = sampler.sample(uncertainties, budget=budget, random_seed=random_seed)
        assert np.sum(pi) <= budget
        assert np.nansum(xi) <= budget
