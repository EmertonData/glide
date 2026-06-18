from unittest.mock import patch

import numpy as np
import pytest

import glide.samplers.clustered as clustered_module
from glide.samplers import UniformClusteredSampler


@pytest.fixture
def sampler() -> UniformClusteredSampler:
    return UniformClusteredSampler()


@pytest.fixture
def clusters() -> np.ndarray:
    return np.array(["A", "A", "B", "B", "C", "C"], dtype=object)


# --- sample ---


def test_sample_delegates_to_validation(sampler, clusters):
    with (
        patch.object(clustered_module, "_validate_is_integer") as mock_validate_is_integer,
        patch.object(clustered_module, "_validate_strictly_positive") as mock_validate_strictly_positive,
        patch.object(clustered_module, "_validate_bounds") as mock_validate_bounds,
    ):
        sampler.sample(clusters, n_clusters=2, random_seed=0)

        mock_validate_is_integer.assert_called_once_with(2, "n_clusters")
        mock_validate_strictly_positive.assert_called_once_with(2, "n_clusters")
        mock_validate_bounds.assert_called_once_with(
            2,
            "n_clusters",
            upper=3,
            error_message="'n_clusters' must not exceed the number of unique clusters; "
            "got n_clusters=2 but there are only 3 unique clusters.",
        )


def test_sample_known_output(sampler, clusters):
    xi = sampler.sample(clusters, n_clusters=2, random_seed=0)

    expected_xi = np.array([0, 0, 1, 1, 1, 1])
    assert isinstance(xi, np.ndarray)
    np.testing.assert_array_equal(xi, expected_xi)


def test_sample_is_reproducible(sampler, clusters):
    xi1 = sampler.sample(clusters, n_clusters=2, random_seed=42)
    xi2 = sampler.sample(clusters, n_clusters=2, random_seed=42)

    np.testing.assert_array_equal(xi1, xi2)


def test_sample_different_seed_results_differ(sampler, clusters):
    xi1 = sampler.sample(clusters, n_clusters=2, random_seed=0)
    xi2 = sampler.sample(clusters, n_clusters=2, random_seed=1)

    assert not np.array_equal(xi1, xi2)
