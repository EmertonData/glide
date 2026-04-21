import numpy as np
import pytest

from glide.simulators.binary import generate_binary_dataset


def test_generate_binary_dataset_structure_and_counts():
    y_true, y_proxy = generate_binary_dataset(n=1, N=2, random_seed=0)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_proxy, np.ndarray)
    assert len(y_true) == 3
    assert len(y_proxy) == 3
    assert np.sum(~np.isnan(y_true)) == 1
    assert np.sum(~np.isnan(y_proxy)) == 3


def test_generate_binary_dataset_invalid_true_mean_raises():
    with pytest.raises(ValueError, match=r"true_mean must be in \(0, 1\), got 1\.5"):
        generate_binary_dataset(n=1, N=1, true_mean=1.5)


def test_generate_binary_dataset_invalid_proxy_mean_raises():
    with pytest.raises(ValueError, match=r"proxy_mean must be in \(0, 1\), got 0"):
        generate_binary_dataset(n=1, N=1, proxy_mean=0.0)


def test_generate_binary_dataset_impossible_correlation_raises():
    with pytest.raises(
        ValueError,
        match=r"Impossible combination of true_mean=0\.7, proxy_mean=0\.6, and correlation=0\.95",
    ):
        generate_binary_dataset(n=1, N=9, true_mean=0.7, proxy_mean=0.6, correlation=0.95)


def test_generate_binary_dataset_reproducibility():
    y_true1, y_proxy1 = generate_binary_dataset(n=1, N=2, random_seed=7)
    y_true2, y_proxy2 = generate_binary_dataset(n=1, N=2, random_seed=7)
    np.testing.assert_allclose(y_true1, y_true2, equal_nan=True)
    np.testing.assert_allclose(y_proxy1, y_proxy2)
