from unittest.mock import patch

import numpy as np
import pytest

import glide.simulators.batched_clustered_binary as batched_clustered_binary_module
from glide.simulators import generate_batched_clustered_binary_dataset


@pytest.fixture
def dataset_params():
    return {
        "n_samples": [4, 3, 2],
        "n_clusters": [2, 3, 2],
        "true_mean": [0.6, 0.6, 0.6],
        "proxy_mean": [0.5, 0.5, 0.5],
        "correlation": [0.7, 0.7, 0.7],
    }


def test_generate_batched_clustered_binary_dataset_structure_and_counts(dataset_params):
    y_true, y_proxy, batches, clusters = generate_batched_clustered_binary_dataset(**dataset_params, random_seed=0)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_proxy, np.ndarray)
    assert isinstance(batches, np.ndarray)
    assert isinstance(clusters, np.ndarray)
    np.testing.assert_allclose(y_true, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(y_proxy, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(batches, [0, 0, 0, 0, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(clusters, [0, 1, 1, 0, 3, 2, 4, 6, 5])


def test_generate_batched_clustered_binary_dataset_clusters_dtype_is_integer(dataset_params):
    _, _, _, clusters = generate_batched_clustered_binary_dataset(**dataset_params, random_seed=0)
    assert np.issubdtype(clusters.dtype, np.integer)


def test_generate_batched_clustered_binary_dataset_cluster_ids_unique_across_batches(dataset_params):
    _, _, batches, clusters = generate_batched_clustered_binary_dataset(**dataset_params, random_seed=0)
    cluster_ids_per_batch = [np.unique(clusters[batches == batch_id]) for batch_id in range(3)]
    np.testing.assert_array_equal(cluster_ids_per_batch[0], [0, 1])
    np.testing.assert_array_equal(cluster_ids_per_batch[1], [2, 3, 4])
    np.testing.assert_array_equal(cluster_ids_per_batch[2], [5, 6])
    assert np.intersect1d(cluster_ids_per_batch[0], cluster_ids_per_batch[1]).size == 0
    assert np.intersect1d(cluster_ids_per_batch[0], cluster_ids_per_batch[2]).size == 0
    assert np.intersect1d(cluster_ids_per_batch[1], cluster_ids_per_batch[2]).size == 0


def test_generate_batched_clustered_binary_dataset_default_within_cluster_diversity(dataset_params):
    default_output = generate_batched_clustered_binary_dataset(**dataset_params, random_seed=0)
    explicit_output = generate_batched_clustered_binary_dataset(
        **dataset_params, within_cluster_diversity=[0.9, 0.9, 0.9], random_seed=0
    )
    for default_array, explicit_array in zip(default_output, explicit_output):
        np.testing.assert_array_equal(default_array, explicit_array)


def test_generate_batched_clustered_binary_dataset_delegates_to_clustered_generator(dataset_params):
    within_cluster_diversity = [0.5, 0.6, 0.7]
    with patch.object(
        batched_clustered_binary_module,
        "generate_clustered_binary_dataset",
        side_effect=[
            (np.array([1.0, 0.0]), np.array([1.0, 1.0]), np.array([0, 1])),
            (np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([0, 1])),
            (np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([1, 0])),
        ],
    ) as mock_generate_clustered_binary_dataset:
        y_true, y_proxy, batches, clusters = generate_batched_clustered_binary_dataset(
            **dataset_params, within_cluster_diversity=within_cluster_diversity
        )

    calls = mock_generate_clustered_binary_dataset.call_args_list
    assert len(calls) == 3
    for batch_id, call in enumerate(calls):
        assert call.kwargs["n_samples"] == dataset_params["n_samples"][batch_id]
        assert call.kwargs["n_clusters"] == dataset_params["n_clusters"][batch_id]
        assert call.kwargs["true_mean"] == dataset_params["true_mean"][batch_id]
        assert call.kwargs["proxy_mean"] == dataset_params["proxy_mean"][batch_id]
        assert call.kwargs["correlation"] == dataset_params["correlation"][batch_id]
        assert call.kwargs["within_cluster_diversity"] == within_cluster_diversity[batch_id]
    np.testing.assert_array_equal(clusters, [0, 1, 2, 3, 6, 5])
    np.testing.assert_array_equal(batches, [0, 0, 1, 1, 2, 2])
    np.testing.assert_allclose(y_true, [1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(y_proxy, [1.0, 1.0, 0.0, 1.0, 0.0, 0.0])


def test_generate_batched_clustered_binary_dataset_delegates_validation(dataset_params):
    n_samples = dataset_params["n_samples"]
    with (
        patch.object(batched_clustered_binary_module, "_validate_non_empty") as mock_validate_non_empty,
        patch.object(batched_clustered_binary_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
    ):
        generate_batched_clustered_binary_dataset(**dataset_params)

        mock_validate_non_empty.assert_called_once_with(n_samples, "n_samples")

        mock_validate_equal_lengths.assert_called_once()
        assert mock_validate_equal_lengths.call_args[0][0] is n_samples
        assert mock_validate_equal_lengths.call_args[0][1] is dataset_params["n_clusters"]
        assert mock_validate_equal_lengths.call_args[0][2] is dataset_params["true_mean"]
        assert mock_validate_equal_lengths.call_args[0][3] is dataset_params["proxy_mean"]
        assert mock_validate_equal_lengths.call_args[0][4] is dataset_params["correlation"]
        assert mock_validate_equal_lengths.call_args[0][5] == [0.9, 0.9, 0.9]
        expected_names = [
            "n_samples",
            "n_clusters",
            "true_mean",
            "proxy_mean",
            "correlation",
            "within_cluster_diversity",
        ]
        assert mock_validate_equal_lengths.call_args[1]["names"] == expected_names


def test_generate_batched_clustered_binary_dataset_reproducibility(dataset_params):
    output1 = generate_batched_clustered_binary_dataset(**dataset_params, random_seed=42)
    output2 = generate_batched_clustered_binary_dataset(**dataset_params, random_seed=42)
    for array1, array2 in zip(output1, output2):
        np.testing.assert_array_equal(array1, array2)


def test_generate_batched_clustered_binary_dataset_different_seed_results_differ(dataset_params):
    dataset_params["n_samples"] = [10, 10, 10]
    y_true1, y_proxy1, _, clusters1 = generate_batched_clustered_binary_dataset(**dataset_params, random_seed=0)
    y_true2, y_proxy2, _, clusters2 = generate_batched_clustered_binary_dataset(**dataset_params, random_seed=1)
    assert (
        not np.array_equal(y_true1, y_true2)
        or not np.array_equal(y_proxy1, y_proxy2)
        or not np.array_equal(clusters1, clusters2)
    )


def test_generate_batched_clustered_binary_dataset_mismatched_lengths(dataset_params):
    dataset_params["n_clusters"] = [2, 3]
    with pytest.raises(ValueError, match="n_clusters"):
        generate_batched_clustered_binary_dataset(**dataset_params)


def test_generate_batched_clustered_binary_dataset_empty_batches():
    with pytest.raises(ValueError, match="n_samples"):
        generate_batched_clustered_binary_dataset(
            n_samples=[], n_clusters=[], true_mean=[], proxy_mean=[], correlation=[]
        )
