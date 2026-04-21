import numpy as np
import pytest

from glide.simulators.gaussian import generate_gaussian_dataset


def test_generate_gaussian_dataset_structure_and_counts():
    y_true, y_proxy = generate_gaussian_dataset(n_labeled=1, n_unlabeled=2, random_seed=0)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_proxy, np.ndarray)
    assert len(y_true) == 3
    assert len(y_proxy) == 3
    np.testing.assert_allclose(y_true, np.array([0.82573022, np.nan, np.nan]), equal_nan=True)
    np.testing.assert_allclose(y_proxy, np.array([0.76352425, 0.17291449, 1.32929515]))


def test_generate_gaussian_dataset_invalid_positive_correlation_raises():
    with pytest.raises(ValueError, match="Correlation should be between -1 and 1"):
        generate_gaussian_dataset(n_labeled=1, n_unlabeled=1, correlation=1.5)


def test_generate_gaussian_dataset_invalid_negative_correlation_raises():
    with pytest.raises(ValueError, match="Correlation should be between -1 and 1"):
        generate_gaussian_dataset(n_labeled=1, n_unlabeled=1, correlation=-1.5)


def test_generate_gaussian_dataset_reproducibility():
    y_true1, y_proxy1 = generate_gaussian_dataset(n_labeled=1, n_unlabeled=2, random_seed=7)
    y_true2, y_proxy2 = generate_gaussian_dataset(n_labeled=1, n_unlabeled=2, random_seed=7)
    np.testing.assert_allclose(y_true1, y_true2, equal_nan=True)
    np.testing.assert_allclose(y_proxy1, y_proxy2)
