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

    dataset = Dataset([{"uncertainty": 1 + i % 10} for i in range(n_records)])
    uncertainties = dataset.to_numpy(fields=["uncertainty"])[:, 0]
    uncertainties = np.minimum(budget * (uncertainties / uncertainties.sum()), 1)

    xi_sums = np.array(
        [
            sampler.sample(dataset, uncertainty_field="uncertainty", budget=budget, seed=seed)["xi"].sum()
            for seed in range(n_trials)
        ]
    )

    expected_std_of_mean = np.sqrt((uncertainties * (1 - uncertainties)).sum() / n_trials)

    assert np.mean(xi_sums) == pytest.approx(budget, abs=3 * expected_std_of_mean)


def test_expected_sum_xi_equals_sum_clipped_pi(sampler):
    """E[sum(xi)] ≈ sum(clipped_pi) < budget when high-weight records are clipped.

    When one record dominates, its raw probability exceeds 1 and is clipped.
    The expected selection count then equals sum(clipped_pi), which is strictly
    less than budget.
    """
    n_trials = 500
    budget = 2

    # One dominant record: raw_pi ≈ 1.9998, clipped to 1.0.
    # Other record:        raw_pi ≈ 0.0002, not clipped.
    # sum(clipped_pi) ≈ 1.0002 < budget=2.
    dataset = Dataset([{"uncertainty": 0.001}, {"uncertainty": 10.0}])

    # Compute the clipped probabilities analytically.
    uncertainties = np.array([0.001, 10.0])
    clipped_pi = np.minimum((uncertainties / uncertainties.sum()) * budget, 1.0)
    expected_sum_xi = clipped_pi.sum()
    assert expected_sum_xi < budget  # confirm clipping is active

    xi_sums = np.array(
        [
            sampler.sample(dataset, uncertainty_field="uncertainty", budget=budget, seed=seed)["xi"].sum()
            for seed in range(n_trials)
        ]
    )

    # Variance = sum(pi_i * (1 - pi_i)); std of mean = sqrt(variance / n_trials).
    variance = np.sum(clipped_pi * (1 - clipped_pi))
    expected_std_of_mean = np.sqrt(variance / n_trials)

    assert np.mean(xi_sums) == pytest.approx(expected_sum_xi, abs=3 * expected_std_of_mean)
