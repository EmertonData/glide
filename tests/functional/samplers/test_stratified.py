import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.simulated_datasets import generate_gaussian_dataset
from glide.estimators.stratified_ppi import StratifiedPPIMeanEstimator
from glide.samplers.stratified import StratifiedSampler


@pytest.fixture
def sampler() -> StratifiedSampler:
    return StratifiedSampler()


def test_neyman_reduces_ci_vs_proportional(sampler):
    random_seed = 42
    n_labeled, n_unlabeled = 30, 50

    # Stratum A: low proxy variance
    labeled_a, unlabeled_a = generate_gaussian_dataset(
        n_labeled, n_unlabeled, true_mean=0.7, true_std=0.1, random_seed=random_seed
    )
    records_a = [{**r, "group": "A"} for r in (labeled_a + unlabeled_a)]

    # Stratum B: high proxy variance
    labeled_b, unlabeled_b = generate_gaussian_dataset(
        n_labeled, n_unlabeled, true_mean=0.4, true_std=1.2, random_seed=random_seed + 1
    )
    records_b = [{**r, "group": "B"} for r in (labeled_b + unlabeled_b)]

    full_dataset = Dataset(records_a + records_b)
    budget = 20

    # Sample with both strategies, which adds 'xi' and 'pi' fields
    proportional_sampled = sampler.sample(
        full_dataset, "y_proxy", "group", budget, strategy="proportional", random_seed=random_seed
    )
    neyman_sampled = sampler.sample(
        full_dataset, "y_proxy", "group", budget, strategy="neyman", random_seed=random_seed
    )

    # Build datasets: keep labeled records selected by sampler (xi=1) + all unlabeled
    def make_dataset(sampled_ds):
        labeled = [r for r in sampled_ds if "y_true" in r and r["xi"] == 1]
        unlabeled = [r for r in sampled_ds if "y_true" not in r]
        return Dataset(labeled + unlabeled)

    proportional_dataset = make_dataset(proportional_sampled)
    neyman_dataset = make_dataset(neyman_sampled)

    # Estimate via StratifiedPPIMeanEstimator
    estimator = StratifiedPPIMeanEstimator()
    proportional_result = estimator.estimate(proportional_dataset, "y_true", "y_proxy", "group")
    neyman_result = estimator.estimate(neyman_dataset, "y_true", "y_proxy", "group")

    proportional_width = (
        proportional_result.confidence_interval.upper_bound - proportional_result.confidence_interval.lower_bound
    )
    neyman_width = neyman_result.confidence_interval.upper_bound - neyman_result.confidence_interval.lower_bound

    assert neyman_width <= proportional_width + 0.2


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
                    "y_proxy": np.sin(stratum_idx + i * 0.5),
                }
            )

    dataset = Dataset(records)
    result = sampler.sample(dataset, "y_proxy", "group", budget, strategy="proportional", random_seed=0)

    # With equal-sized strata, proportional allocation gives uniform pi across all records
    total_size = len(records)
    expected_pi = budget / total_size

    for record in result:
        assert np.isclose(record["pi"], expected_pi)
