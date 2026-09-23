from unittest.mock import patch

import numpy as np

import glide.simulators.batched_stratified_binary as batched_stratified_binary_module
from glide.simulators import generate_batched_stratified_binary_dataset


def test_generate_batched_stratified_binary_dataset_structure_and_counts():
    y_true, y_proxy, batches, groups = generate_batched_stratified_binary_dataset(
        n_samples=[[3, 3], [3, 3]],
        true_mean=[[0.6, 0.8], [0.6, 0.8]],
        proxy_mean=[[0.5, 0.7], [0.5, 0.7]],
        correlation=[[0.7, 0.7], [0.7, 0.7]],
        random_seed=0,
    )
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_proxy, np.ndarray)
    assert isinstance(batches, np.ndarray)
    assert isinstance(groups, np.ndarray)
    assert len(y_true) == 12
    assert len(y_proxy) == 12
    assert len(batches) == 12
    assert len(groups) == 12
    y_true_expected = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    y_proxy_expected = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    np.testing.assert_allclose(y_true, y_true_expected)
    np.testing.assert_allclose(y_proxy, y_proxy_expected)
    np.testing.assert_array_equal(batches, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(groups, [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])


def test_generate_batched_stratified_binary_dataset_different_strata_per_batch():
    y_true, y_proxy, batches, groups = generate_batched_stratified_binary_dataset(
        n_samples=[[2], [2, 3]],
        true_mean=[[0.6], [0.6, 0.8]],
        proxy_mean=[[0.5], [0.5, 0.7]],
        correlation=[[0.7], [0.7, 0.7]],
        random_seed=0,
    )
    assert len(y_true) == 7
    assert len(y_proxy) == 7
    np.testing.assert_array_equal(batches, [0, 0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(groups, [0, 0, 0, 0, 1, 1, 1])


def test_generate_batched_stratified_binary_dataset_delegates_validation():
    n_samples = [[3, 3], [3, 3]]
    true_mean = [[0.5, 0.6], [0.5, 0.6]]
    proxy_mean = [[0.5, 0.6], [0.5, 0.6]]
    correlation = [[0.8, 0.8], [0.8, 0.8]]

    with (
        patch.object(batched_stratified_binary_module, "_validate_non_empty") as mock_validate_non_empty,
        patch.object(batched_stratified_binary_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
    ):
        generate_batched_stratified_binary_dataset(
            n_samples=n_samples,
            true_mean=true_mean,
            proxy_mean=proxy_mean,
            correlation=correlation,
        )

        mock_validate_non_empty.assert_called_once_with(n_samples, "n_samples")

        mock_validate_equal_lengths.assert_called_once()
        assert mock_validate_equal_lengths.call_args[0][0] is n_samples
        assert mock_validate_equal_lengths.call_args[0][1] is true_mean
        assert mock_validate_equal_lengths.call_args[0][2] is proxy_mean
        assert mock_validate_equal_lengths.call_args[0][3] is correlation
        expected_names = ["n_samples", "true_mean", "proxy_mean", "correlation"]
        assert mock_validate_equal_lengths.call_args[1]["names"] == expected_names


def test_generate_batched_stratified_binary_dataset_reproducibility():
    y_true1, y_proxy1, batches1, groups1 = generate_batched_stratified_binary_dataset(
        n_samples=[[3, 3], [3, 3]],
        true_mean=[[0.6, 0.8], [0.6, 0.8]],
        proxy_mean=[[0.5, 0.7], [0.5, 0.7]],
        correlation=[[0.7, 0.7], [0.7, 0.7]],
        random_seed=42,
    )
    y_true2, y_proxy2, batches2, groups2 = generate_batched_stratified_binary_dataset(
        n_samples=[[3, 3], [3, 3]],
        true_mean=[[0.6, 0.8], [0.6, 0.8]],
        proxy_mean=[[0.5, 0.7], [0.5, 0.7]],
        correlation=[[0.7, 0.7], [0.7, 0.7]],
        random_seed=42,
    )
    np.testing.assert_allclose(y_true1, y_true2)
    np.testing.assert_allclose(y_proxy1, y_proxy2)
    np.testing.assert_array_equal(batches1, batches2)
    np.testing.assert_array_equal(groups1, groups2)


def test_generate_batched_stratified_binary_dataset_different_seed_results_differ():
    y_true1, y_proxy1, _, _ = generate_batched_stratified_binary_dataset(
        n_samples=[[5, 5], [5, 5]],
        true_mean=[[0.6, 0.8], [0.6, 0.8]],
        proxy_mean=[[0.5, 0.7], [0.5, 0.7]],
        correlation=[[0.7, 0.7], [0.7, 0.7]],
        random_seed=0,
    )
    y_true2, y_proxy2, _, _ = generate_batched_stratified_binary_dataset(
        n_samples=[[5, 5], [5, 5]],
        true_mean=[[0.6, 0.8], [0.6, 0.8]],
        proxy_mean=[[0.5, 0.7], [0.5, 0.7]],
        correlation=[[0.7, 0.7], [0.7, 0.7]],
        random_seed=1,
    )
    assert not np.array_equal(y_true1, y_true2) or not np.array_equal(y_proxy1, y_proxy2)
