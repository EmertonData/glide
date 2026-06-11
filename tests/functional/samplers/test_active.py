import numpy as np
import pytest

from glide.samplers import ActiveSampler


@pytest.fixture
def sampler() -> ActiveSampler:
    return ActiveSampler()


def test_sample_never_exceeds_budget(sampler):
    n_total = 50
    n_samples = 10
    n_trials = 500

    uncertainties = np.ones(n_total)

    for random_seed in range(n_trials):
        pi, xi = sampler.sample(uncertainties, n_samples=n_samples, random_seed=random_seed)
        assert np.sum(pi) <= n_samples
        assert np.nansum(xi) <= n_samples
