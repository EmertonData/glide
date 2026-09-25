from unittest.mock import patch

import numpy as np
import pytest

import glide.engines.stratified_ppi as stratified_ppi_engine_module
from glide.engines.stratified_ppi import StratifiedPPIMeanEngine


@pytest.fixture
def y_true():
    return np.array([5.0, 6.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan])


@pytest.fixture
def y_proxy():
    return np.array([4.9, 6.1, 5.2, 6.1, 4.9, 6.1, 5.2, 6.1])


@pytest.fixture
def groups():
    return np.array(["A", "A", "A", "A", "B", "B", "B", "B"])


@pytest.fixture
def engine():
    return StratifiedPPIMeanEngine()


@pytest.fixture
def dataset(engine, y_true, y_proxy, groups):
    return engine.preprocess(y_true, y_proxy, groups)


# --- preprocess ---


def test_preprocess_delegates_to_stratified_core(engine, y_true, y_proxy, groups):
    sentinel_dataset = object()
    with patch.object(stratified_ppi_engine_module, "_preprocess") as mock_preprocess:
        mock_preprocess.return_value = sentinel_dataset
        dataset = engine.preprocess(y_true, y_proxy, groups)

        mock_preprocess.assert_called_once()
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][1], y_proxy)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][2], groups)
        assert dataset is sentinel_dataset


# --- fit_tuning_parameter ---


def test_fit_tuning_parameter_power_tuning_true(engine, dataset):
    tuning_parameter = engine.fit_tuning_parameter(dataset, power_tuning=True)
    assert tuning_parameter["A"] == pytest.approx(0.784, abs=1e-3)
    assert tuning_parameter["B"] == pytest.approx(0.784, abs=1e-3)


def test_fit_tuning_parameter_power_tuning_false(engine, dataset):
    tuning_parameter = engine.fit_tuning_parameter(dataset, power_tuning=False)
    assert tuning_parameter["A"] == pytest.approx(1.0)
    assert tuning_parameter["B"] == pytest.approx(1.0)


def test_fit_tuning_parameter_wraps_error_with_stratum_identifier(engine):
    y_true = np.array([1.0, 2.0, np.nan, np.nan, 1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    dataset = engine.preprocess(y_true, y_proxy, groups)

    expected_message = "Proxy labels have zero variance; cannot estimate the tuning parameter. (stratum 'B')."
    with pytest.raises(ValueError, match=r"^Proxy labels have zero variance") as exc_info:
        engine.fit_tuning_parameter(dataset, power_tuning=True)
    assert str(exc_info.value) == expected_message


# --- compute_mean_and_std ---


def test_compute_mean_and_std(engine, dataset):
    tuning_parameter = {"A": 0.784, "B": 0.784}
    mean, std = engine.compute_mean_and_std(dataset, tuning_parameter)
    assert mean == pytest.approx(5.617, abs=1e-3)
    assert std == pytest.approx(0.250, abs=1e-3)


def test_compute_mean_and_std_missing_stratum_defaults_to_untuned(engine, dataset):
    mean_with_missing_key, std_with_missing_key = engine.compute_mean_and_std(dataset, {"A": 0.784})
    mean_with_explicit_one, std_with_explicit_one = engine.compute_mean_and_std(dataset, {"A": 0.784, "B": 1.0})

    assert mean_with_missing_key == pytest.approx(mean_with_explicit_one)
    assert std_with_missing_key == pytest.approx(std_with_explicit_one)
