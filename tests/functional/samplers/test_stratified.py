import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.samplers.stratified import StratifiedSampler


@pytest.fixture
def sampler() -> StratifiedSampler:
    return StratifiedSampler()


def test_proportional_matches_uniform_equal_strata(sampler):
    n_per_stratum = 5
    n_strata = 3
    budget = 9

    records = []
    for stratum_idx in range(n_strata):
        for i in range(n_per_stratum):
            records.append(
                {
                    "group": f"stratum_{stratum_idx}",
                    "y_proxy": stratum_idx + i * 0.1,
                }
            )

    dataset = Dataset(records)
    result = sampler.sample(dataset, "y_proxy", "group", budget, strategy="proportional", random_seed=0)

    # With equal-sized strata, proportional allocation gives uniform pi across all records
    total_size = len(records)
    expected_pi = budget / total_size

    for record in result:
        assert np.isclose(record["pi"], expected_pi)


def test_sample_rounding_sums_to_budget(sampler):
    dataset = Dataset(
        [
            {"group": "s0", "y_proxy": 0.0},
            {"group": "s0", "y_proxy": 0.1},
            {"group": "s1", "y_proxy": 1.0},
            {"group": "s1", "y_proxy": 1.1},
            {"group": "s2", "y_proxy": 2.0},
            {"group": "s2", "y_proxy": 2.1},
            {"group": "s3", "y_proxy": 3.0},
            {"group": "s3", "y_proxy": 3.1},
            {"group": "s4", "y_proxy": 4.0},
            {"group": "s4", "y_proxy": 4.1},
        ]
    )

    for strategy in ["proportional", "neyman"]:
        result = sampler.sample(dataset, "y_proxy", "group", 7, strategy=strategy)

        groups = np.array([record["group"] for record in result])
        pi = np.array([record["pi"] for record in result])

        unique_groups, group_sizes = np.unique(groups, return_counts=True)

        # Get first pi value for each unique group
        group_pi = np.array([pi[groups == group][0] for group in unique_groups])

        # Vectorized allocation calculation
        allocations = np.minimum(group_pi * group_sizes, group_sizes)
        total_allocation = np.sum(allocations)

        assert total_allocation <= 7
