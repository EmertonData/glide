import pytest
from glide.core.dataset import Dataset
from glide.core.simulated_datasets import generate_binary_dataset


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
