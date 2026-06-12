from unittest.mock import call, patch

import numpy as np
import pytest

from glide.simulators import generate_multi_binary_dataset


def test_generate_multi_binary_dataset_structure_and_counts():
    y_true, y_proxies = generate_multi_binary_dataset(4, 0.7, [0.6, 0.5], [0.5, 0.4], random_seed=0)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_proxies, np.ndarray)
    assert y_true.shape == (4,)
    assert y_proxies.shape == (4, 2)
    assert np.isin(y_true, [0.0, 1.0]).all()
    assert np.isin(y_proxies, [0.0, 1.0]).all()


def test_generate_multi_binary_dataset_delegates_to_validation():
    with (
        patch("glide.simulators.multi_binary._validate_equal_lengths") as mock_validate_equal_lengths,
        patch("glide.simulators.multi_binary._validate_bounds") as mock_validate_bounds,
    ):
        generate_multi_binary_dataset(2, 0.7, [0.6, 0.5], [0.5, 0.4], random_seed=0)
    mock_validate_equal_lengths.assert_called_once()
    np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], np.array([0.6, 0.5]))
    np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], np.array([0.5, 0.4]))
    assert mock_validate_equal_lengths.call_args[1]["names"] == ["proxy_means", "correlations"]
    mock_validate_bounds.assert_has_calls(
        [
            call(0.7, "true_mean", lower=0, upper=1, left_inclusive=False, right_inclusive=False),
            call(0.6, "proxy_means[0]", lower=0, upper=1, left_inclusive=False, right_inclusive=False),
            call(0.5, "proxy_means[1]", lower=0, upper=1, left_inclusive=False, right_inclusive=False),
        ]
    )


def test_generate_multi_binary_dataset_impossible_correlation_raises():
    with pytest.raises(ValueError, match=r"proxy 1:"):
        generate_multi_binary_dataset(4, 0.7, [0.6, 0.6], [0.5, 0.95])


def test_generate_multi_binary_dataset_reproducibility():
    y_true1, y_proxies1 = generate_multi_binary_dataset(4, 0.7, [0.6, 0.5], [0.5, 0.4], random_seed=7)
    y_true2, y_proxies2 = generate_multi_binary_dataset(4, 0.7, [0.6, 0.5], [0.5, 0.4], random_seed=7)
    np.testing.assert_array_equal(y_true1, y_true2)
    np.testing.assert_array_equal(y_proxies1, y_proxies2)


def test_generate_multi_binary_dataset_different_seeds_differ():
    y_true1, y_proxies1 = generate_multi_binary_dataset(10, 0.7, [0.6], [0.5], random_seed=0)
    y_true2, y_proxies2 = generate_multi_binary_dataset(10, 0.7, [0.6], [0.5], random_seed=1)
    assert (not np.array_equal(y_true1, y_true2)) or (not np.array_equal(y_proxies1, y_proxies2))
