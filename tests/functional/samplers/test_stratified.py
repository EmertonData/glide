import numpy as np
import pytest

from glide.samplers import StratifiedSampler


@pytest.fixture
def sampler() -> StratifiedSampler:
    return StratifiedSampler()


def test_proportional_matches_uniform_equal_strata(sampler):
    n_per_stratum = 5
    n_strata = 3
    budget = 9

    groups = np.repeat(np.arange(n_strata), n_per_stratum)
    sample_indices = np.tile(np.arange(n_per_stratum), n_strata)
    y_proxy = groups + sample_indices * 0.1

    pi, _ = sampler.sample(y_proxy, groups, budget, strategy="proportional", random_seed=0)

    # With equal-sized strata, proportional allocation gives uniform pi across all samples
    total_size = len(y_proxy)
    expected_pi = budget / total_size

    np.testing.assert_allclose(pi, expected_pi)


def test_sample_rounding_sums_to_budget(sampler):
    y_proxy = np.array([0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1])
    groups = np.array(["s0", "s0", "s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"], dtype=object)

    for strategy in ["proportional", "neyman"]:
        pi, _ = sampler.sample(y_proxy, groups, 7, strategy=strategy)

        unique_groups, group_sizes = np.unique(groups, return_counts=True)

        # Get first pi value for each unique group
        group_pi = np.array([pi[groups == group][0] for group in unique_groups])

        # Vectorized allocation calculation
        allocations = np.minimum(group_pi * group_sizes, group_sizes)
        total_allocation = np.sum(allocations)

        assert total_allocation <= 7
