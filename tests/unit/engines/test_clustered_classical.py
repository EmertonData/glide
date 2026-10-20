from unittest.mock import patch

import numpy as np
import pytest

import glide.engines.clustered_classical as clustered_classical_engine_module
from glide.engines.classical import ClassicalMeanEngine
from glide.engines.clustered_classical import ClusteredClassicalMeanEngine


@pytest.fixture
def y():
    return np.array([5.0, 5.0, 7.0, 7.0])


@pytest.fixture
def clusters():
    return np.array(["A", "A", "B", "B"])


@pytest.fixture
def engine():
    return ClusteredClassicalMeanEngine()


@pytest.fixture
def dataset(engine, y, clusters):
    return engine.preprocess(y, clusters)


# --- __init__ ---


def test_init_sets_classical_engine(engine):
    assert isinstance(engine._classical_engine, ClassicalMeanEngine)


# --- preprocess ---


def test_preprocess_delegates_to_validation(engine, y, clusters):
    with (
        patch.object(clustered_classical_engine_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(clustered_classical_engine_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(clustered_classical_engine_module, "_validate_bounds") as mock_validate_bounds,
    ):
        engine.preprocess(y, clusters)

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], clusters)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y", "clusters"]}

        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], clusters)
        assert mock_validate_has_no_nan.call_args[0][1] == "clusters"

        mock_validate_bounds.assert_called_once_with(
            2,
            "n_valid_clusters",
            lower=2,
            error_message="Need at least 2 clusters with non-NaN observations; got 2.",
        )


def test_preprocess_delegates_to_compute_cluster_means(engine):
    y = np.array([2.0, np.nan, 4.0, 6.0])
    clusters = np.array(["A", "A", "B", "C"])
    sentinel_cluster_means = np.array([1.0, 2.0, 3.0])
    with patch.object(clustered_classical_engine_module, "_compute_cluster_means") as mock_compute_cluster_means:
        mock_compute_cluster_means.return_value = sentinel_cluster_means
        cluster_means = engine.preprocess(y, clusters)

        mock_compute_cluster_means.assert_called_once()
        np.testing.assert_array_equal(mock_compute_cluster_means.call_args[0][0], np.array([2.0, 4.0, 6.0]))
        np.testing.assert_array_equal(mock_compute_cluster_means.call_args[0][1], np.array([0, 1, 2]))
        assert cluster_means is sentinel_cluster_means


def test_preprocess_known_result(engine):
    y = np.array([2.0, 4.0, np.nan, 6.0, 10.0])
    clusters = np.array(["A", "A", "B", "C", "C"])
    cluster_means = engine.preprocess(y, clusters)
    np.testing.assert_allclose(cluster_means, np.array([3.0, 8.0]))


def test_preprocess_unequal_lengths(engine):
    with pytest.raises(ValueError, match="'y' and 'clusters' must have the same length; got 3 and 2."):
        engine.preprocess(np.array([1.0, 2.0, 3.0]), np.array(["A", "B"]))


def test_preprocess_nan_clusters(engine):
    with pytest.raises(ValueError, match="'clusters' contains NaN values."):
        engine.preprocess(np.array([1.0, 2.0]), np.array([0.0, np.nan]))


def test_preprocess_none_clusters(engine):
    with pytest.raises(ValueError, match="'clusters' contains None values."):
        engine.preprocess(np.array([1.0, 2.0]), np.array(["A", None], dtype=object))


def test_preprocess_too_few_valid_clusters(engine):
    y = np.array([1.0, 2.0, np.nan])
    clusters = np.array(["A", "A", "B"])
    with pytest.raises(ValueError, match="Need at least 2 clusters with non-NaN observations; got 1."):
        engine.preprocess(y, clusters)


# --- fit_tuning_parameter ---


def test_fit_tuning_parameter_delegates_to_classical_engine(engine, dataset):
    with patch.object(engine._classical_engine, "fit_tuning_parameter") as mock_fit_tuning_parameter:
        mock_fit_tuning_parameter.return_value = None
        tuning_parameter = engine.fit_tuning_parameter(dataset, power_tuning=False)

        mock_fit_tuning_parameter.assert_called_once()
        np.testing.assert_array_equal(mock_fit_tuning_parameter.call_args[0][0], dataset)
        assert mock_fit_tuning_parameter.call_args.kwargs == {"power_tuning": False}
        assert tuning_parameter is None


# --- compute_mean_and_std ---


def test_compute_mean_and_std_delegates_to_classical_engine(engine, dataset):
    with patch.object(engine._classical_engine, "compute_mean_and_std") as mock_compute_mean_and_std:
        mock_compute_mean_and_std.return_value = (1.5, 0.25)
        mean, std = engine.compute_mean_and_std(dataset, tuning_parameter=None)

        mock_compute_mean_and_std.assert_called_once()
        np.testing.assert_array_equal(mock_compute_mean_and_std.call_args[0][0], dataset)
        assert mock_compute_mean_and_std.call_args[0][1] is None
        assert (mean, std) == (1.5, 0.25)


def test_compute_mean_and_std_known_result(engine, dataset):
    mean, std = engine.compute_mean_and_std(dataset, tuning_parameter=None)
    assert mean == pytest.approx(6.0, abs=1e-12)
    assert std == pytest.approx(1.0, abs=1e-12)
