import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.simulated_datasets import generate_binary_dataset


def test_dataset_empty():
    dataset = Dataset()
    assert dataset == []


def test_dataset_iadd():
    dataset = Dataset()
    dataset += [{"a": 0}]
    assert dataset == [{"a": 0}]


def test_dataset_imul():
    dataset = Dataset([{"a": 0}])
    dataset *= 2
    assert dataset == [{"a": 0}, {"a": 0}]


def test_dataset_records():
    dataset = Dataset([{"a": 0}, {"b": 1}])
    assert dataset.records == [{"a": 0}, {"b": 1}]


RECORDS = [
    {"human": 0, "llm": 0},
    {"llm": 1},
]


def test_to_numpy_human_then_llm():
    result = Dataset(RECORDS).to_numpy(fields=["human", "llm"])
    expected = np.array([[0, 0], [np.nan, 1]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_to_numpy_llm_then_human():
    result = Dataset(RECORDS).to_numpy(fields=["llm", "human"])
    expected = np.array([[0, 0], [1, np.nan]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_to_numpy_single_field():
    result = Dataset(RECORDS).to_numpy(fields=["llm"])
    expected = np.array([[0], [1]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_to_numpy_unknown_field_raises():
    with pytest.raises(ValueError):
        Dataset(RECORDS).to_numpy(fields=["unknown"])


# --- generate_binary_dataset ---


def test_generate_binary_dataset_returns_dataset():
    ds = generate_binary_dataset(n=1, N=1, random_seed=0)
    assert isinstance(ds, Dataset)


def test_generate_binary_dataset_counts():
    ds = generate_binary_dataset(n=1, N=2, random_seed=0)
    labeled = [r for r in ds if "y_true" in r]
    unlabeled = [r for r in ds if "y_true" not in r]
    assert len(ds) == 3
    assert len(labeled) == 1
    assert len(unlabeled) == 2


def test_generate_binary_dataset_record_structure():
    ds = generate_binary_dataset(n=2, N=2, random_seed=0)
    for record in ds:
        assert "y_proxy" in record
        assert record["y_proxy"] in (0, 1)
        assert record.get("y_true", 0) in (0, 1)


def test_generate_binary_dataset_impossible_correlation_raises():
    with pytest.raises(AssertionError):
        generate_binary_dataset(n=1, N=9, true_mean=0.7, proxy_mean=0.6, correlation=0.95)


def test_generate_binary_dataset_reproducibility():
    ds1 = generate_binary_dataset(n=1, N=2, random_seed=7)
    ds2 = generate_binary_dataset(n=1, N=2, random_seed=7)
    assert ds1 == ds2
