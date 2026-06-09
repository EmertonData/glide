from unittest.mock import call, patch

import numpy as np

import glide.simulators.clustered_binary as clustered_binary_module
from glide.simulators import generate_clustered_binary_dataset


def test_generate_clustered_binary_dataset_structure_and_counts():
    seed = np.random.SeedSequence(42)
    y_true, y_proxy, clusters = generate_clustered_binary_dataset(n_total=4, n_clusters=2, random_seed=seed)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_proxy, np.ndarray)
    assert isinstance(clusters, np.ndarray)
    assert len(y_true) == 4
    assert len(y_proxy) == 4
    assert len(clusters) == 4
    assert clusters.dtype == np.int64
    assert np.isin(y_true, [0.0, 1.0]).all()
    assert np.isin(y_proxy, [0.0, 1.0]).all()
    np.testing.assert_array_equal(np.unique(clusters), np.arange(2))


def test_generate_clustered_binary_dataset_delegates():
    mock_return = (np.array([1.0, 0.0, 1.0, 1.0]), np.array([1.0, 0.0, 0.0, 1.0]))
    with (
        patch.object(clustered_binary_module, "_validate_bounds") as mock_validate_bounds,
        patch.object(
            clustered_binary_module, "generate_binary_dataset", return_value=mock_return
        ) as mock_generate_binary_dataset,
    ):
        generate_clustered_binary_dataset(
            n_total=4, n_clusters=2, true_mean=0.7, proxy_mean=0.6, correlation=0.8, random_seed=0
        )
        mock_validate_bounds.assert_has_calls(
            [
                call(2, "n_clusters", lower=2, error_message="'n_clusters' must be >= 2; got 2."),
                call(
                    4,
                    "n_total",
                    lower=2,
                    error_message="'n_total' must be >= 'n_clusters'; got n_total=4 and n_clusters=2.",
                ),
                call(0.9, "within_cluster_diversity", lower=0, upper=1),
            ]
        )
        mock_generate_binary_dataset.assert_called_once()
        call_kwargs = mock_generate_binary_dataset.call_args.kwargs
        assert call_kwargs["n_total"] == 4
        assert call_kwargs["true_mean"] == 0.7
        assert call_kwargs["proxy_mean"] == 0.6
        assert call_kwargs["correlation"] == 0.8
        assert isinstance(call_kwargs["random_seed"], np.random.SeedSequence)


def test_generate_clustered_binary_dataset_reproducibility():
    y_true1, y_proxy1, clusters1 = generate_clustered_binary_dataset(n_total=6, n_clusters=2, random_seed=7)
    y_true2, y_proxy2, clusters2 = generate_clustered_binary_dataset(n_total=6, n_clusters=2, random_seed=7)
    np.testing.assert_allclose(y_true1, y_true2)
    np.testing.assert_allclose(y_proxy1, y_proxy2)
    np.testing.assert_array_equal(clusters1, clusters2)


def test_generate_clustered_binary_dataset_different_seed_results_differ():
    y_true1, y_proxy1, clusters1 = generate_clustered_binary_dataset(n_total=10, n_clusters=2, random_seed=0)
    y_true2, y_proxy2, clusters2 = generate_clustered_binary_dataset(n_total=10, n_clusters=2, random_seed=1)
    assert (
        not np.array_equal(y_true1, y_true2)
        or not np.array_equal(y_proxy1, y_proxy2)
        or not np.array_equal(clusters1, clusters2)
    )
