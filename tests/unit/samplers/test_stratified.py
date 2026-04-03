import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.samplers.stratified import StratifiedSampler


@pytest.fixture
def sampler() -> StratifiedSampler:
    return StratifiedSampler()


@pytest.fixture
def dataset() -> Dataset:
    return Dataset(
        [
            {"group": "A", "y_proxy": 0.5},
            {"group": "A", "y_proxy": 0.6},
            {"group": "B", "y_proxy": 0.7},
            {"group": "B", "y_proxy": 0.8},
        ]
    )


@pytest.fixture
def y_proxy() -> np.ndarray:
    return np.array([0.5, 0.6, 0.7, 0.8])


@pytest.fixture
def groups() -> np.ndarray:
    return np.array(["A", "A", "B", "B"], dtype=object)


@pytest.fixture
def dataset_varied() -> Dataset:
    return Dataset(
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


# --- _preprocess ---


def test_preprocess_groups_preserved(sampler):
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.5},
            {"group": "A", "y_proxy": 0.6},
            {"group": "B", "y_proxy": 0.7},
            {"group": "B", "y_proxy": 0.8},
        ]
    )
    y_proxy, groups = sampler._preprocess(dataset, "y_proxy", "group")

    assert np.array_equal(groups, np.array(["A", "A", "B", "B"], dtype=object))


def test_preprocess_raises_on_empty_dataset(sampler):
    dataset = Dataset([])

    with pytest.raises(ValueError, match="empty"):
        sampler._preprocess(dataset, "y_proxy", "group")


def test_preprocess_raises_on_unknown_y_proxy_field(sampler):
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.5},
            {"group": "A", "y_proxy": 0.6},
        ]
    )

    with pytest.raises(ValueError):
        sampler._preprocess(dataset, "nonexistent_field", "group")


def test_preprocess_raises_on_unknown_groups_field(sampler):
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.5},
            {"group": "A", "y_proxy": 0.6},
        ]
    )

    with pytest.raises((ValueError, KeyError)):
        sampler._preprocess(dataset, "y_proxy", "nonexistent_group_field")


def test_preprocess_raises_on_stratum_size_less_than_two(sampler):
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.5},
            {"group": "A", "y_proxy": 0.6},
            {"group": "B", "y_proxy": 0.7},
        ]
    )

    with pytest.raises(ValueError, match="fewer than 2 records"):
        sampler._preprocess(dataset, "y_proxy", "group")


def test_preprocess_raises_on_nan_proxy(sampler):
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.5},
            {"group": "A", "y_proxy": np.nan},
        ]
    )

    with pytest.raises(ValueError, match="NaN"):
        sampler._preprocess(dataset, "y_proxy", "group")


def test_preprocess_raises_on_zero_variance(sampler):
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 5.0},
            {"group": "A", "y_proxy": 5.0},
            {"group": "B", "y_proxy": 5.0},
            {"group": "B", "y_proxy": 5.0},
        ]
    )

    with pytest.raises(ValueError, match="zero variance"):
        sampler._preprocess(dataset, "y_proxy", "group")


def test_preprocess_raises_on_all_strata_zero_variance(sampler):
    dataset = Dataset(
        [
            {"group": "A", "y_proxy": 0.5},
            {"group": "A", "y_proxy": 0.5},
            {"group": "B", "y_proxy": 0.9},
            {"group": "B", "y_proxy": 0.9},
        ]
    )

    with pytest.raises(ValueError, match="All strata have zero variance"):
        sampler._preprocess(dataset, "y_proxy", "group")


# --- _proportional_allocation ---


def test_proportional_allocation_proportional_to_N_h(sampler, y_proxy, groups):
    budget = 4
    total = len(groups)

    allocation = sampler._proportional_allocation(groups, budget)

    expected_ratio = budget / total
    for stratum_id, n_h in allocation.items():
        N_h = (groups == stratum_id).sum()
        actual_ratio = n_h / N_h
        assert actual_ratio == pytest.approx(expected_ratio, abs=0.01)


# --- sample ---


def test_sample_returns_dataset_with_valid_properties(sampler, dataset):
    result = sampler.sample(dataset, "y_proxy", "group", 2, random_seed=0)

    assert isinstance(result, Dataset)
    assert len(result) == len(dataset)
    assert all("pi" in record and "xi" in record for record in result)
    pi_values = result["pi"]
    assert np.all(pi_values > 0)
    assert np.all(pi_values <= 1)
    xi_values = result["xi"]
    assert set(xi_values).issubset({0.0, 1.0})


def test_sample_invalid_budget_zero(sampler, dataset):
    with pytest.raises(ValueError):
        sampler.sample(dataset, "y_proxy", "group", 0)


def test_sample_invalid_budget_negative(sampler, dataset):
    with pytest.raises(ValueError):
        sampler.sample(dataset, "y_proxy", "group", -1)


def test_sample_invalid_budget_float(sampler, dataset):
    with pytest.raises((ValueError, TypeError)):
        sampler.sample(dataset, "y_proxy", "group", 1.5)


def test_sample_invalid_budget_boolean(sampler, dataset):
    with pytest.raises((ValueError, TypeError)):
        sampler.sample(dataset, "y_proxy", "group", True)


def test_sample_budget_exceeds_dataset_length(sampler, dataset):
    with pytest.raises(ValueError):
        sampler.sample(dataset, "y_proxy", "group", len(dataset) + 1)


def test_sample_default_strategy_is_neyman(sampler, dataset_varied):
    budget = 8

    default_result = sampler.sample(dataset_varied, "y_proxy", "group", budget)
    neyman_result = sampler.sample(dataset_varied, "y_proxy", "group", budget, strategy="neyman")

    assert [r["pi"] for r in default_result] == [r["pi"] for r in neyman_result]


def test_sample_neyman_strategy(sampler, dataset_varied):
    result = sampler.sample(dataset_varied, "y_proxy", "group", 8, strategy="neyman")

    pi_a = result[0]["pi"]
    pi_b = result[4]["pi"]
    assert pi_b > pi_a


def test_sample_invalid_strategy_raises(sampler, dataset):
    with pytest.raises(ValueError, match="Unknown strategy"):
        sampler.sample(dataset, "y_proxy", "group", 2, strategy="unknown")


def test_sample_is_reproducible(sampler, dataset):
    result1 = sampler.sample(dataset, "y_proxy", "group", 2, random_seed=42)
    result2 = sampler.sample(dataset, "y_proxy", "group", 2, random_seed=42)

    np.testing.assert_array_equal(result1["xi"], result2["xi"])


def test_sample_seed_defaults_to_none_without_exception(sampler, dataset):
    result = sampler.sample(dataset, "y_proxy", "group", 2)

    assert isinstance(result, Dataset)


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
def test_sample_rounding_sums_to_budget(sampler, n_records, n_strata, budget):
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
        result = sampler.sample(dataset, "y_proxy", "group", budget, strategy=strategy)
        group_pi_values = {}
        for record in result:
            group = record["group"]
            if group not in group_pi_values:
                group_pi_values[group] = record["pi"]

        group_sizes = {}
        for record in result:
            group = record["group"]
            group_sizes[group] = group_sizes.get(group, 0) + 1

        total_allocation = sum(
            min(pi * group_sizes[group], group_sizes[group]) for group, pi in group_pi_values.items()
        )
        assert total_allocation <= budget
