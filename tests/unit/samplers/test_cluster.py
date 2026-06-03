from unittest.mock import patch

import numpy as np
import pytest

import glide.samplers.cluster as cluster_module
from glide.samplers import ClusterSampler


@pytest.fixture
def sampler() -> ClusterSampler:
    return ClusterSampler()


@pytest.fixture
def clusters() -> np.ndarray:
    return np.array(["A", "A", "B", "B", "C", "C"], dtype=object)


# --- _validate ---


def test_validate_delegates_to_validation(sampler, clusters):
    with (
        patch.object(cluster_module, "_validate_is_integer") as mock_validate_is_integer,
        patch.object(cluster_module, "_validate_strictly_positive") as mock_validate_strictly_positive,
        patch.object(cluster_module, "_validate_literal") as mock_validate_literal,
    ):
        sampler._validate(clusters, 2, "uniform")

        mock_validate_is_integer.assert_called_once_with(2, "n_clusters")
        mock_validate_strictly_positive.assert_called_once_with(2, "n_clusters")
        mock_validate_literal.assert_called_once_with("uniform", "strategy", ["uniform", "proportional"])


def test_validate_n_clusters_exceeds_unique_clusters(sampler, clusters):
    with pytest.raises(ValueError, match="'n_clusters' must not exceed the number of unique clusters"):
        sampler._validate(clusters, 4, "uniform")


# --- sample ---


def test_sample_returns_valid_array(sampler, clusters):
    xi = sampler.sample(clusters, n_clusters=2, random_seed=0)
    assert isinstance(xi, np.ndarray)
    assert xi.shape == (len(clusters),)
    assert np.isin(xi, [0, 1]).all()


def test_sample_default_strategy_is_uniform(sampler, clusters):
    xi_default = sampler.sample(clusters, n_clusters=2, random_seed=0)
    xi_uniform = sampler.sample(clusters, n_clusters=2, strategy="uniform", random_seed=0)

    np.testing.assert_array_equal(xi_default, xi_uniform)


def test_sample_uniform_strategy(sampler, clusters):
    xi = sampler.sample(clusters, n_clusters=2, strategy="uniform", random_seed=0)

    expected_xi = np.array([0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_proportional_strategy(sampler):
    clusters = np.array(["A", "B", "B", "C", "C", "C"], dtype=object)
    xi = sampler.sample(clusters, n_clusters=2, strategy="proportional", random_seed=1)

    expected_xi = np.array([1, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_all_clusters_selected(sampler, clusters):
    xi = sampler.sample(clusters, n_clusters=3, random_seed=0)

    np.testing.assert_array_equal(xi, np.ones(len(clusters), dtype=int))


def test_sample_single_cluster(sampler):
    clusters = np.array(["X", "X", "X"], dtype=object)
    xi = sampler.sample(clusters, n_clusters=1, random_seed=0)

    np.testing.assert_array_equal(xi, np.ones(3, dtype=int))


def test_sample_is_reproducible(sampler, clusters):
    xi1 = sampler.sample(clusters, n_clusters=2, random_seed=42)
    xi2 = sampler.sample(clusters, n_clusters=2, random_seed=42)

    np.testing.assert_array_equal(xi1, xi2)
