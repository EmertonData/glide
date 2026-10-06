from unittest.mock import patch

import numpy as np
import pytest

import glide.engines.clustered_ppi as clustered_ppi_engine_module
from glide.engines.clustered_ppi import ClusteredPPIMeanEngine
from glide.engines.ppi import PPIMeanEngine


@pytest.fixture
def y_true():
    return np.array([4.0, np.nan, 6.0, np.nan])


@pytest.fixture
def y_proxy():
    return np.array([2.0, 4.0, 6.0, 6.0])


@pytest.fixture
def clusters():
    return np.array(["A", "B", "C", "D"])


@pytest.fixture
def engine():
    return ClusteredPPIMeanEngine()


@pytest.fixture
def dataset(engine, y_true, y_proxy, clusters):
    return engine.preprocess(y_true, y_proxy, clusters)


# --- __init__ ---


def test_init_sets_ppi_engine(engine):
    assert isinstance(engine._ppi_engine, PPIMeanEngine)


# --- preprocess ---


def test_preprocess_delegates_to_clustered_core(engine, y_true, y_proxy, clusters):
    sentinel_dataset = object()
    with patch.object(clustered_ppi_engine_module, "_preprocess") as mock_preprocess:
        mock_preprocess.return_value = sentinel_dataset
        dataset = engine.preprocess(y_true, y_proxy, clusters)

        mock_preprocess.assert_called_once()
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][1], y_proxy)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][2], clusters)
        assert dataset is sentinel_dataset


# --- fit_tuning_parameter ---


def test_fit_tuning_parameter_delegates_to_ppi_engine(engine, dataset):
    with patch.object(engine._ppi_engine, "fit_tuning_parameter") as mock_fit_tuning_parameter:
        mock_fit_tuning_parameter.return_value = 0.5
        tuning_parameter = engine.fit_tuning_parameter(dataset, power_tuning=False)

        mock_fit_tuning_parameter.assert_called_once()
        for received, expected in zip(mock_fit_tuning_parameter.call_args[0][0], dataset):
            np.testing.assert_array_equal(received, expected)
        assert mock_fit_tuning_parameter.call_args.kwargs == {"power_tuning": False}
        assert tuning_parameter == 0.5


# --- compute_mean_and_std ---


def test_compute_mean_and_std_delegates_to_ppi_engine(engine, dataset):
    with patch.object(engine._ppi_engine, "compute_mean_and_std") as mock_compute_mean_and_std:
        mock_compute_mean_and_std.return_value = (1.5, 0.25)
        mean, std = engine.compute_mean_and_std(dataset, tuning_parameter=0.5)

        mock_compute_mean_and_std.assert_called_once()
        for received, expected in zip(mock_compute_mean_and_std.call_args[0][0], dataset):
            np.testing.assert_array_equal(received, expected)
        assert mock_compute_mean_and_std.call_args[0][1] == 0.5
        assert (mean, std) == (1.5, 0.25)
