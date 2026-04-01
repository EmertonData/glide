import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.samplers.stratified import StratifiedSampler


@pytest.fixture
def sampler() -> StratifiedSampler:
    return StratifiedSampler()


def test_preprocess_groups_preserved(sampler):
    """groups array contains the exact stratum identifiers from the records."""
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.5},
            {"group": "A", "y_proxy": 0.6},
            {"group": "B", "y_proxy": 0.7},
        ]
    )
    y_proxy, groups = sampler._preprocess(dataset, "group", "y_proxy")

    assert np.array_equal(groups, np.array(["A", "A", "B"], dtype=object))


def test_proportional_sums_to_budget(sampler):
    """Proportional allocation: Σ n_h ≤ budget."""
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.5},
            {"group": "A", "y_proxy": 0.6},
            {"group": "B", "y_proxy": 0.7},
            {"group": "B", "y_proxy": 0.8},
        ]
    )
    y_proxy, groups = sampler._preprocess(dataset, "group", "y_proxy")
    budget = 4

    allocation = sampler._proportional_allocation(y_proxy, groups, budget)

    assert sum(allocation.values()) <= budget


def test_proportional_proportional_to_N_h(sampler):
    """n_h/N_h ≈ n/N for all strata (within rounding error)."""
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.5},
            {"group": "A", "y_proxy": 0.6},
            {"group": "B", "y_proxy": 0.7},
            {"group": "B", "y_proxy": 0.8},
        ]
    )
    y_proxy, groups = sampler._preprocess(dataset, "group", "y_proxy")
    budget = 4
    total = len(groups)

    allocation = sampler._proportional_allocation(y_proxy, groups, budget)

    # Expected ratio: n/N = budget/total = 1.0
    expected_ratio = budget / total
    for stratum_id, n_h in allocation.items():
        N_h = (groups == stratum_id).sum()
        actual_ratio = n_h / N_h
        # Within rounding tolerance
        assert abs(actual_ratio - expected_ratio) < 0.5


def test_neyman_sums_to_budget(sampler):
    """Neyman allocation: Σ n_h ≤ budget."""
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.4},
            {"group": "A", "y_proxy": 0.6},
            {"group": "B", "y_proxy": 0.1},
            {"group": "B", "y_proxy": 0.9},
            {"group": "C", "y_proxy": 0.3},
            {"group": "C", "y_proxy": 0.7},
            {"group": "D", "y_proxy": 0.2},
            {"group": "D", "y_proxy": 0.8},
        ]
    )
    y_proxy, groups = sampler._preprocess(dataset, "group", "y_proxy")
    budget = 8

    allocation = sampler._neyman_allocation(y_proxy, groups, budget)

    assert sum(allocation.values()) <= budget


def test_neyman_assigns_more_to_high_variance_stratum(sampler):
    """Two strata with equal N_h but different S_h: higher-variance stratum gets more."""
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.45},
            {"group": "A", "y_proxy": 0.50},
            {"group": "A", "y_proxy": 0.55},
            {"group": "A", "y_proxy": 0.60},
            {"group": "B", "y_proxy": 0.0},
            {"group": "B", "y_proxy": 0.33},
            {"group": "B", "y_proxy": 0.67},
            {"group": "B", "y_proxy": 1.0},
        ]
    )
    y_proxy, groups = sampler._preprocess(dataset, "group", "y_proxy")
    budget = 8

    allocation = sampler._neyman_allocation(y_proxy, groups, budget)

    # B has higher variance than A
    assert allocation["B"] > allocation["A"]


def test_neyman_zero_variance_fallback(sampler):
    """If all S_h == 0, result equals proportional allocation."""
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 5.0},
            {"group": "A", "y_proxy": 5.0},
            {"group": "B", "y_proxy": 3.0},
            {"group": "B", "y_proxy": 3.0},
            {"group": "C", "y_proxy": 7.0},
            {"group": "C", "y_proxy": 7.0},
            {"group": "D", "y_proxy": 4.0},
            {"group": "D", "y_proxy": 4.0},
        ]
    )
    y_proxy, groups = sampler._preprocess(dataset, "group", "y_proxy")
    budget = 8

    neyman_result = sampler._neyman_allocation(y_proxy, groups, budget)
    proportional_result = sampler._proportional_allocation(y_proxy, groups, budget)

    assert neyman_result == proportional_result


def test_single_stratum_returns_budget(sampler):
    """One stratum → n_h = budget."""
    dataset = Dataset(
        [
            {"group": "X", "y_proxy": 0.5},
            {"group": "X", "y_proxy": 0.6},
            {"group": "X", "y_proxy": 0.7},
        ]
    )
    budget = 3
    result = sampler.sample(dataset, "group", "y_proxy", budget)

    assert result[0]["n_h"] == budget


def test_allocate_budget_default_mode_is_neyman(sampler):
    """Calling without strategy produces same result as strategy='neyman'."""
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.45},
            {"group": "A", "y_proxy": 0.50},
            {"group": "A", "y_proxy": 0.55},
            {"group": "A", "y_proxy": 0.60},
            {"group": "B", "y_proxy": 0.0},
            {"group": "B", "y_proxy": 0.33},
            {"group": "B", "y_proxy": 0.67},
            {"group": "B", "y_proxy": 1.0},
        ]
    )
    budget = 8

    default_result = sampler.sample(dataset, "group", "y_proxy", budget)
    neyman_result = sampler.sample(dataset, "group", "y_proxy", budget, strategy="neyman")

    # Check that n_h values match
    assert [r["n_h"] for r in default_result] == [r["n_h"] for r in neyman_result]


def test_allocate_budget_neyman_mode(sampler):
    """strategy='neyman' dispatches to Neyman allocator."""
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.45},
            {"group": "A", "y_proxy": 0.50},
            {"group": "A", "y_proxy": 0.55},
            {"group": "A", "y_proxy": 0.60},
            {"group": "B", "y_proxy": 0.0},
            {"group": "B", "y_proxy": 0.33},
            {"group": "B", "y_proxy": 0.67},
            {"group": "B", "y_proxy": 1.0},
        ]
    )
    budget = 8

    result = sampler.sample(dataset, "group", "y_proxy", budget, strategy="neyman")

    # Extract per-stratum allocations
    n_h_a = result[0]["n_h"]
    n_h_b = result[4]["n_h"]  # First record in group B
    assert n_h_b > n_h_a


def test_allocate_budget_invalid_mode_raises(sampler):
    """Passing an unknown strategy string raises ValueError."""
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.5},
            {"group": "B", "y_proxy": 0.7},
        ]
    )

    with pytest.raises(ValueError, match="Unknown strategy"):
        sampler.sample(dataset, "group", "y_proxy", 2, strategy="unknown")


@pytest.mark.parametrize(
    "n_records,n_strata,budget",
    [
        (4, 2, 3),
        (4, 2, 4),
        (6, 3, 4),
        (6, 3, 6),
        (10, 5, 7),
    ],
)
def test_rounding_sums_to_budget_within_limit(sampler, n_records, n_strata, budget):
    """Multiple budget/stratum combinations — Σ n_h ≤ budget."""
    records = []
    for stratum_idx in range(n_strata):
        n_per_stratum = n_records // n_strata
        for i in range(n_per_stratum):
            records.append(
                {
                    "group": f"s{stratum_idx}",
                    "y_proxy": float(stratum_idx + i * 0.1),
                }
            )
    dataset = Dataset(records)

    for strategy in ["proportional", "neyman"]:
        result = sampler.sample(dataset, "group", "y_proxy", budget, strategy=strategy)
        # Compute sum of unique n_h values per stratum
        unique_strata = {}
        for record in result:
            group = record["group"]
            if group not in unique_strata:
                unique_strata[group] = record["n_h"]
        assert sum(unique_strata.values()) <= budget
