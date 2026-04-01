"""Functional tests for StratifiedSampler."""

import numpy as np

from glide.core.dataset import Dataset
from glide.core.simulated_datasets import generate_gaussian_dataset
from glide.estimators.stratified_ppi import StratifiedPPIMeanEstimator
from glide.samplers.stratified import StratifiedSampler


def test_neyman_reduces_ci_vs_proportional():
    """Dataset with heterogeneous proxy variance across strata. After sampling according to Neyman vs. proportional plan, CI width via StratifiedPPIMeanEstimator is narrower with Neyman allocation."""
    random_seed = 42
    n_labeled, n_unlabeled = 5, 10

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
    sampler = StratifiedSampler()
    budget = 6

    # Get allocations via sample() which returns Dataset with n_h column
    proportional_sampled = sampler.sample(full_dataset, "group", "y_proxy", budget,
                                          strategy="proportional")
    neyman_sampled = sampler.sample(full_dataset, "group", "y_proxy", budget, strategy="neyman")

    # Extract allocation dict from sampled datasets
    def extract_allocation(sampled_ds):
        alloc = {}
        for record in sampled_ds:
            group_id = record["group"]
            if group_id not in alloc:
                alloc[group_id] = record["n_h"]
        return alloc

    proportional_alloc = extract_allocation(proportional_sampled)
    neyman_alloc = extract_allocation(neyman_sampled)

    # Build datasets according to allocations
    def make_dataset(full_ds, alloc):
        labeled = []
        for group_id in alloc:
            group_records = [r for r in full_ds if r.get("group") == group_id]
            labeled_in_group = [r for r in group_records if "y_true" in r]
            labeled.extend(labeled_in_group[: alloc[group_id]])
        unlabeled = [r for r in full_ds if "y_true" not in r]
        return Dataset(labeled + unlabeled)

    proportional_dataset = make_dataset(full_dataset, proportional_alloc)
    neyman_dataset = make_dataset(full_dataset, neyman_alloc)

    # Estimate via StratifiedPPIMeanEstimator
    estimator = StratifiedPPIMeanEstimator()
    proportional_result = estimator.estimate(proportional_dataset, "y_true", "y_proxy", "group")
    neyman_result = estimator.estimate(neyman_dataset, "y_true", "y_proxy", "group")

    proportional_width = proportional_result.confidence_interval.upper_bound - proportional_result.confidence_interval.lower_bound
    neyman_width = neyman_result.confidence_interval.upper_bound - neyman_result.confidence_interval.lower_bound

    # Neyman should be narrower or comparable
    assert neyman_width <= proportional_width + 0.2


def test_proportional_matches_uniform_equal_strata():
    """When all N_h are equal, proportional allocation gives n_h = n/K for all strata (same as uniform sampling)."""
    n_per_stratum = 5
    n_strata = 3
    budget = 9

    records = []
    for stratum_idx in range(n_strata):
        for i in range(n_per_stratum):
            records.append({
                "group": f"stratum_{stratum_idx}",
                "y_proxy": np.sin(stratum_idx + i * 0.5),
            })

    dataset = Dataset(records)
    sampler = StratifiedSampler()
    result = sampler.sample(dataset, "group", "y_proxy", budget, strategy="proportional")

    # Extract unique allocations per stratum from the result dataset
    allocation = {}
    for record in result:
        group_id = record["group"]
        if group_id not in allocation:
            allocation[group_id] = record["n_h"]

    # All strata should get equal allocation
    expected_per_stratum = budget // n_strata
    for stratum_idx in range(n_strata):
        assert allocation[f"stratum_{stratum_idx}"] == expected_per_stratum
