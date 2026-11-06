from unittest.mock import patch

import numpy as np
import pytest

import glide.simulators.batched_oracle_binary as batched_oracle_binary_module
from glide.simulators import generate_batched_binary_dataset_with_oracle_sampling


@pytest.fixture
def dataset_params():
    return {
        "n_samples": [3, 2],
        "true_mean": [0.6, 0.8],
        "proxy_mean": [0.5, 0.7],
        "correlation": [0.7, 0.7],
    }


def test_generate_batched_binary_dataset_with_oracle_sampling_structure_and_counts(dataset_params):
    y_true_oracle, y_proxy, uncertainty, batches = generate_batched_binary_dataset_with_oracle_sampling(
        **dataset_params, random_seed=0
    )
    y_true_oracle_expected = [0.0, 1.0, 1.0, 1.0, 1.0]
    y_proxy_expected = [0.0, 0.0, 1.0, 1.0, 1.0]
    uncertainty_expected = [0.3340, 0.4194, 0.3663, 0.3404, 0.3663]
    np.testing.assert_allclose(y_true_oracle, y_true_oracle_expected)
    np.testing.assert_allclose(y_proxy, y_proxy_expected)
    np.testing.assert_allclose(uncertainty, uncertainty_expected, atol=1e-4)
    np.testing.assert_array_equal(batches, [0, 0, 0, 1, 1])


def test_generate_batched_binary_dataset_with_oracle_sampling_delegates_validation(dataset_params):
    n_samples = dataset_params["n_samples"]
    true_mean = dataset_params["true_mean"]
    proxy_mean = dataset_params["proxy_mean"]
    correlation = dataset_params["correlation"]

    with (
        patch.object(batched_oracle_binary_module, "_validate_non_empty") as mock_validate_non_empty,
        patch.object(batched_oracle_binary_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
    ):
        generate_batched_binary_dataset_with_oracle_sampling(**dataset_params)

        mock_validate_non_empty.assert_called_once_with(n_samples, "n_samples")

        mock_validate_equal_lengths.assert_called_once()
        assert mock_validate_equal_lengths.call_args[0][0] is n_samples
        assert mock_validate_equal_lengths.call_args[0][1] is true_mean
        assert mock_validate_equal_lengths.call_args[0][2] is proxy_mean
        assert mock_validate_equal_lengths.call_args[0][3] is correlation
        expected_names = ["n_samples", "true_mean", "proxy_mean", "correlation"]
        assert mock_validate_equal_lengths.call_args[1]["names"] == expected_names


def test_generate_batched_binary_dataset_with_oracle_sampling_reproducibility(dataset_params):
    first = generate_batched_binary_dataset_with_oracle_sampling(**dataset_params, random_seed=42)
    second = generate_batched_binary_dataset_with_oracle_sampling(**dataset_params, random_seed=42)
    for first_array, second_array in zip(first, second, strict=True):
        np.testing.assert_allclose(first_array, second_array)


def test_generate_batched_binary_dataset_with_oracle_sampling_different_seed_results_differ():
    params = {"true_mean": [0.6, 0.8], "proxy_mean": [0.5, 0.7], "correlation": [0.7, 0.7]}
    uncertainty_first = generate_batched_binary_dataset_with_oracle_sampling(n_samples=[5, 5], **params, random_seed=0)[
        2
    ]
    uncertainty_second = generate_batched_binary_dataset_with_oracle_sampling(
        n_samples=[5, 5], **params, random_seed=1
    )[2]
    assert not np.array_equal(uncertainty_first, uncertainty_second)
