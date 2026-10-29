from unittest.mock import patch

import numpy as np
import pytest

import glide.engines.stratified_classical as stratified_classical_engine_module
from glide.engines.classical import ClassicalMeanEngine
from glide.engines.stratified_classical import StratifiedClassicalMeanEngine


@pytest.fixture
def y():
    return np.array([1.0, 3.0, np.nan, 5.0, 7.0])


@pytest.fixture
def groups():
    return np.array(["A", "A", "A", "B", "B"])


@pytest.fixture
def engine():
    return StratifiedClassicalMeanEngine()


@pytest.fixture
def dataset(engine, y, groups):
    return engine._preprocess(y, groups)


# --- __init__ ---


def test_init_sets_classical_engine(engine):
    assert isinstance(engine._classical_engine, ClassicalMeanEngine)


# --- _preprocess ---


def test_preprocess_delegates_to_validation(engine, y, groups):
    with (
        patch.object(stratified_classical_engine_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(stratified_classical_engine_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(stratified_classical_engine_module, "_validate_min_samples") as mock_validate_min_samples,
    ):
        engine._preprocess(y, groups)

        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], groups)
        assert mock_validate_has_no_nan.call_args[0][1] == "groups"

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], groups)
        assert mock_validate_equal_lengths.call_args.kwargs == {"names": ["y", "groups"]}

        assert len(mock_validate_min_samples.call_args_list) == 2
        np.testing.assert_array_equal(mock_validate_min_samples.call_args_list[0][0][0], np.array([1.0, 3.0]))
        assert mock_validate_min_samples.call_args_list[0][0][1:] == ("y", "A")
        np.testing.assert_array_equal(mock_validate_min_samples.call_args_list[1][0][0], np.array([5.0, 7.0]))
        assert mock_validate_min_samples.call_args_list[1][0][1:] == ("y", "B")


def test_preprocess_valid_output(dataset):
    expected_y_by_stratum = {"A": np.array([1.0, 3.0]), "B": np.array([5.0, 7.0])}
    expected_stratum_weights = {"A": 0.6, "B": 0.4}

    assert dataset.keys() == expected_y_by_stratum.keys()
    for stratum_id, expected_y in expected_y_by_stratum.items():
        y_stratum, stratum_weight = dataset[stratum_id]
        np.testing.assert_array_equal(y_stratum, expected_y)
        assert stratum_weight == pytest.approx(expected_stratum_weights[stratum_id], abs=1e-12)


def test_preprocess_custom_stratum_weights(engine, y, groups):
    dataset = engine._preprocess(y, groups, stratum_weights=np.array([0.8, 0.2]))

    assert dataset["A"][1] == pytest.approx(0.8, abs=1e-12)
    assert dataset["B"][1] == pytest.approx(0.2, abs=1e-12)


# --- _fit_tuning_parameter ---


def test_fit_tuning_parameter_returns_none(engine, dataset):
    assert engine._fit_tuning_parameter(dataset, power_tuning=True) is None
    assert engine._fit_tuning_parameter(dataset, power_tuning=False) is None


# --- _compute_mean_and_std ---


def test_compute_mean_and_std(engine, dataset):
    mean, std = engine._compute_mean_and_std(dataset, None)
    assert mean == pytest.approx(3.6)
    assert std == pytest.approx(np.sqrt(0.52))
