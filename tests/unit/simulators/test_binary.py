from unittest.mock import call, patch

import numpy as np
import pytest

from glide.simulators import generate_binary_dataset


def test_generate_binary_dataset_structure_and_counts():
    y_true, y_proxy = generate_binary_dataset(n_total=3, random_seed=0)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_proxy, np.ndarray)
    assert y_true.shape == (3,)
    assert y_proxy.shape == (3,)
    assert np.isin(y_true, [0.0, 1.0]).all()
    assert np.isin(y_proxy, [0.0, 1.0]).all()


def test_generate_binary_dataset_delegates_validation():
    with patch("glide.simulators.binary._validate_bounds") as mock_validate_bounds:
        generate_binary_dataset(n_total=2, true_mean=0.7, proxy_mean=0.6, correlation=0.8, random_seed=0)
    mock_validate_bounds.assert_has_calls(
        [
            call(0.7, "true_mean", lower=0, upper=1, left_inclusive=False, right_inclusive=False),
            call(0.6, "proxy_mean", lower=0, upper=1, left_inclusive=False, right_inclusive=False),
        ]
    )


def test_generate_binary_dataset_impossible_correlation_raises():
    with pytest.raises(
        ValueError,
        match=r"Impossible combination of 'true_mean'=0\.7, 'proxy_mean'=0\.6, and 'correlation'=0\.95",
    ):
        generate_binary_dataset(n_total=10, true_mean=0.7, proxy_mean=0.6, correlation=0.95)


def test_generate_binary_dataset_reproducibility():
    y_true1, y_proxy1 = generate_binary_dataset(n_total=3, random_seed=7)
    y_true2, y_proxy2 = generate_binary_dataset(n_total=3, random_seed=7)
    np.testing.assert_allclose(y_true1, y_true2)
    np.testing.assert_allclose(y_proxy1, y_proxy2)


def test_generate_binary_dataset_different_seed_results_differ():
    y_true1, y_proxy1 = generate_binary_dataset(n_total=10, random_seed=0)
    y_true2, y_proxy2 = generate_binary_dataset(n_total=10, random_seed=1)
    assert (not np.array_equal(y_true1, y_true2)) or (not np.array_equal(y_proxy1, y_proxy2))
