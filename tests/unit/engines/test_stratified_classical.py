from unittest.mock import patch

import numpy as np
import pytest

import glide.engines.stratified_classical as stratified_classical_engine_module
from glide.engines.classical import ClassicalMeanEngine
from glide.engines.stratified_classical import StratifiedClassicalMeanEngine


@pytest.fixture
def y():
    return np.array([1.0, 3.0, 5.0, 7.0])


@pytest.fixture
def groups():
    return np.array(["A", "A", "B", "B"])


@pytest.fixture
def engine():
    return StratifiedClassicalMeanEngine()


@pytest.fixture
def dataset(engine, y, groups):
    return engine.preprocess(y, groups)


# --- __init__ ---


def test_init_sets_classical_engine(engine):
    assert isinstance(engine._classical_engine, ClassicalMeanEngine)


# --- preprocess ---


def test_preprocess_delegates_to_validation(engine, y, groups):
    with (
        patch.object(stratified_classical_engine_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(stratified_classical_engine_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(stratified_classical_engine_module, "_validate_min_samples") as mock_validate_min_samples,
    ):
        engine.preprocess(y, groups)

        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], groups)
        assert mock_validate_has_no_nan.call_args[0][1] == "groups"

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], groups)
        assert mock_validate_equal_lengths.call_args.kwargs["names"] == ["y", "groups"]

        assert len(mock_validate_min_samples.call_args_list) == 2
        np.testing.assert_array_equal(mock_validate_min_samples.call_args_list[0][0][0], np.array([1.0, 3.0]))
        assert mock_validate_min_samples.call_args_list[0][0][1:] == ("y", "A")
        np.testing.assert_array_equal(mock_validate_min_samples.call_args_list[1][0][0], np.array([5.0, 7.0]))
        assert mock_validate_min_samples.call_args_list[1][0][1:] == ("y", "B")


def test_preprocess_splits_strata(engine, dataset):
    assert list(dataset.keys()) == ["A", "B"]
    np.testing.assert_array_equal(dataset["A"][0], np.array([1.0, 3.0]))
    np.testing.assert_array_equal(dataset["B"][0], np.array([5.0, 7.0]))
    assert dataset["A"][1] == pytest.approx(0.5)
    assert dataset["B"][1] == pytest.approx(0.5)


def test_preprocess_default_weights_are_row_shares_with_nan_rows(engine):
    y = np.array([1.0, 3.0, np.nan, np.nan, 5.0, 7.0])
    groups = np.array(["A", "A", "A", "A", "B", "B"])

    dataset = engine.preprocess(y, groups)

    np.testing.assert_array_equal(dataset["A"][0], np.array([1.0, 3.0]))
    np.testing.assert_array_equal(dataset["B"][0], np.array([5.0, 7.0]))
    assert dataset["A"][1] == pytest.approx(2 / 3)
    assert dataset["B"][1] == pytest.approx(1 / 3)


def test_preprocess_custom_stratum_weights(engine, y, groups):
    dataset = engine.preprocess(y, groups, np.array([0.8, 0.2]))

    assert dataset["A"][1] == pytest.approx(0.8)
    assert dataset["B"][1] == pytest.approx(0.2)


def test_preprocess_unequal_lengths(engine, y):
    with pytest.raises(ValueError, match=r"^'y' and 'groups' must have the same length; got 4 and 3\.$"):
        engine.preprocess(y, np.array(["A", "A", "B"]))


def test_preprocess_stratum_with_only_nan_values(engine):
    y = np.array([1.0, 3.0, np.nan, np.nan])
    groups = np.array(["A", "A", "B", "B"])

    expected_message = "'y' must have at least 2 non-NaN values per stratum; got 0 in stratum 'B'."
    with pytest.raises(ValueError) as exc_info:
        engine.preprocess(y, groups)
    assert str(exc_info.value) == expected_message


# --- fit_tuning_parameter ---


def test_fit_tuning_parameter_returns_none(engine, dataset):
    assert engine.fit_tuning_parameter(dataset, power_tuning=True) is None
    assert engine.fit_tuning_parameter(dataset, power_tuning=False) is None


# --- compute_mean_and_std ---


def test_compute_mean_and_std(engine, dataset):
    mean, std = engine.compute_mean_and_std(dataset, None)
    assert mean == pytest.approx(4.0)
    assert std == pytest.approx(np.sqrt(0.5))


def test_compute_mean_and_std_delegates_to_classical_engine(engine, dataset):
    with patch.object(
        engine._classical_engine, "compute_mean_and_std", wraps=engine._classical_engine.compute_mean_and_std
    ) as mock_compute_mean_and_std:
        engine.compute_mean_and_std(dataset, None)

    assert len(mock_compute_mean_and_std.call_args_list) == 2
    np.testing.assert_array_equal(mock_compute_mean_and_std.call_args_list[0][0][0], np.array([1.0, 3.0]))
    np.testing.assert_array_equal(mock_compute_mean_and_std.call_args_list[1][0][0], np.array([5.0, 7.0]))
    assert mock_compute_mean_and_std.call_args_list[0][0][1] is None
    assert mock_compute_mean_and_std.call_args_list[1][0][1] is None


def test_compute_mean_and_std_with_unequal_row_shares_and_nan_rows(engine):
    y = np.array([1.0, 3.0, np.nan, np.nan, 5.0, 7.0])
    groups = np.array(["A", "A", "A", "A", "B", "B"])
    dataset = engine.preprocess(y, groups)

    mean, std = engine.compute_mean_and_std(dataset, None)

    assert mean == pytest.approx(10 / 3)
    assert std == pytest.approx(np.sqrt(5) / 3)
