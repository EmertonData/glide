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
    return engine._preprocess(y, clusters)


# --- __init__ ---


def test_init_sets_classical_engine(engine):
    assert isinstance(engine._classical_engine, ClassicalMeanEngine)


# --- _preprocess ---


def test_preprocess_delegates_to_validation(engine, y, clusters):
    with (
        patch.object(clustered_classical_engine_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(clustered_classical_engine_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(clustered_classical_engine_module, "_validate_bounds") as mock_validate_bounds,
    ):
        engine._preprocess(y, clusters)

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], clusters)
        assert mock_validate_equal_lengths.call_args.kwargs == {"names": ["y", "clusters"]}

        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], clusters)
        assert mock_validate_has_no_nan.call_args[0][1] == "clusters"

        mock_validate_bounds.assert_called_once_with(
            2,
            "n_valid_clusters",
            lower=2,
            error_message="Need at least 2 clusters with non-NaN observations; got 2.",
        )


def test_preprocess_valid_output(engine, y, clusters):
    cluster_means = engine._preprocess(y, clusters)
    expected = np.array([5.0, 7.0])
    np.testing.assert_allclose(cluster_means, expected)


# --- _fit_tuning_parameter ---


def test_fit_tuning_parameter_returns_none(engine, dataset):
    assert engine._fit_tuning_parameter(dataset, power_tuning=True) is None
    assert engine._fit_tuning_parameter(dataset, power_tuning=False) is None


# --- _compute_mean_and_std ---


def test_compute_mean_and_std_delegates_to_classical_engine(engine, dataset):
    with patch.object(engine._classical_engine, "_compute_mean_and_std") as mock_compute_mean_and_std:
        mock_compute_mean_and_std.return_value = (1.5, 0.25)
        mean, std = engine._compute_mean_and_std(dataset, tuning_parameter=None)

        mock_compute_mean_and_std.assert_called_once()
        np.testing.assert_array_equal(mock_compute_mean_and_std.call_args[0][0], dataset)
        assert mock_compute_mean_and_std.call_args[0][1] is None
        assert (mean, std) == (1.5, 0.25)
