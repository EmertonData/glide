import numpy as np
import pytest

from glide.samplers import ActiveSampler


@pytest.fixture
def sampler() -> ActiveSampler:
    return ActiveSampler()


def test_expected_sum_xi_equals_budget(sampler):
    n_samples = 50
    budget = 10
    n_trials = 500

    uncertainties = np.ones(n_samples)

    n_array = np.array(
        [
            sampler.sample(uncertainties, budget=budget, random_seed=random_seed)[1].sum()
            for random_seed in range(n_trials)
        ]
    )

    assert np.mean(n_array) == pytest.approx(10.238, abs=0.001)
