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
