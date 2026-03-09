import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.simulated import generate_dataset_binary


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


# --- generate_dataset_binary ---


def test_generate_dataset_binary_returns_dataset():
    ds = generate_dataset_binary(n=10, N=20, random_seed=0)
    assert isinstance(ds, Dataset)


def test_generate_dataset_binary_counts():
    ds = generate_dataset_binary(n=100, N=400, random_seed=0)
    labeled = [r for r in ds if "true" in r]
    unlabeled = [r for r in ds if "true" not in r]
    assert len(ds) == 500
    assert len(labeled) == 100
    assert len(unlabeled) == 400


def test_generate_dataset_binary_record_structure():
    ds = generate_dataset_binary(n=50, N=50, random_seed=0)
    for record in ds:
        assert "proxy" in record
        assert record["proxy"] in (0, 1)
    labeled = [r for r in ds if "true" in r]
    for record in labeled:
        assert record["true"] in (0, 1)


def test_generate_dataset_binary_impossible_correlation_raises():
    # correlation=0.95 makes probs[1] = proxy_mean - both_1_prob negative
    with pytest.raises(AssertionError):
        generate_dataset_binary(n=10, N=90, true_mean=0.7, proxy_mean=0.6, correlation=0.95)


def test_generate_dataset_binary_empirical_means():
    ds = generate_dataset_binary(n=500, N=4500, true_mean=0.7, proxy_mean=0.6, random_seed=42)
    labeled = [r for r in ds if "true" in r]
    true_mean = np.mean([r["true"] for r in labeled])
    proxy_mean = np.mean([r["proxy"] for r in ds])
    assert abs(true_mean - 0.7) < 0.03
    assert abs(proxy_mean - 0.6) < 0.03


def test_generate_dataset_binary_reproducibility():
    ds1 = generate_dataset_binary(n=10, N=90, random_seed=7)
    ds2 = generate_dataset_binary(n=10, N=90, random_seed=7)
    assert ds1 == ds2
