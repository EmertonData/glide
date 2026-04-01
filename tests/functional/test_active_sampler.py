import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.samplers.active import ActiveSampler


@pytest.fixture
def sampler() -> ActiveSampler:
    return ActiveSampler()


def test_expected_sum_xi_equals_budget(sampler):
    """E[sum(xi)] ≈ budget when no probability is clipped.

    With uniform uncertainties, pi_i = budget / n for every record (no clipping).
    By linearity of expectation, E[sum(xi)] = sum(pi_i) = budget exactly.
    """
    n_records = 50
    budget = 10
    n_trials = 500

    dataset = Dataset([{"uncertainty": 1} for _ in range(n_records)])

    n_array = np.array(
        [
            sampler.sample(dataset, uncertainty_field="uncertainty", budget=budget, random_seed=random_seed)["xi"].sum()
            for random_seed in range(n_trials)
        ]
    )

    assert np.mean(n_array) == pytest.approx(10.238, abs=0.001)
