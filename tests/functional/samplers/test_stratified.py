import numpy as np
import pytest

from glide.samplers import StratifiedSampler


@pytest.fixture
def sampler() -> StratifiedSampler:
    return StratifiedSampler()


def test_proportional_matches_uniform_equal_strata(sampler):
    n_per_stratum = 5
    n_strata = 3
    n_samples = 9

    groups = np.repeat(np.arange(n_strata), n_per_stratum)
    sample_indices = np.tile(np.arange(n_per_stratum), n_strata)
    y_proxy = groups + sample_indices * 0.1

    xi = sampler.sample(y_proxy, groups, n_samples, strategy="proportional", random_seed=0)

    # With equal-sized strata, proportional allocation gives each stratum the same number of annotations
    expected_per_stratum = n_samples // n_strata
    for stratum_id in np.unique(groups):
        assert xi[groups == stratum_id].sum() == expected_per_stratum


def test_sample_rounding_sums_to_budget(sampler):
    y_proxy = np.array([0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.1] * 2)
    groups = np.array(["s0", "s0", "s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"] * 2, dtype=object)

    for strategy in ["proportional", "neyman"]:
        xi = sampler.sample(y_proxy, groups, 10, strategy=strategy, random_seed=0)

        assert xi.sum() <= 10
