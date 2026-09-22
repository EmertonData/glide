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


def test_preprocess_delegates_to_validation(engine, y_true, y_proxy, groups):
    labeled_mask = np.array([True, True, False, False])
    with (
        patch.object(stratified_ppi_engine_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(stratified_ppi_engine_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(stratified_ppi_engine_module, "_split_labeled_unlabeled") as mock_split_labeled_unlabeled,
        patch.object(stratified_ppi_engine_module, "_validate_sample_sizes") as mock_validate_sample_sizes,
    ):
        mock_split_labeled_unlabeled.side_effect = [
            (np.array([5.0, 6.0]), np.array([4.9, 6.1]), np.array([5.2, 6.1]), labeled_mask),
            (np.array([5.0, 6.0]), np.array([4.9, 6.1]), np.array([5.2, 6.1]), labeled_mask),
        ]
        engine.preprocess(y_true, y_proxy, groups)

        assert mock_validate_has_no_nan.call_count == 3
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args_list[0][0][0], groups)
        assert mock_validate_has_no_nan.call_args_list[0][0][1] == "groups"
        y_proxy_a = y_proxy[groups == "A"]
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args_list[1][0][0], y_proxy_a)
        assert mock_validate_has_no_nan.call_args_list[1][0][1] == "y_proxy"
        y_proxy_b = y_proxy[groups == "B"]
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args_list[2][0][0], y_proxy_b)
        assert mock_validate_has_no_nan.call_args_list[2][0][1] == "y_proxy"

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], y_proxy)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][2], groups)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y_true", "y_proxy", "groups"]}

        assert mock_split_labeled_unlabeled.call_count == 2
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args_list[0][0][0], y_true[groups == "A"])
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args_list[0][0][1], y_proxy[groups == "A"])
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args_list[1][0][0], y_true[groups == "B"])
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args_list[1][0][1], y_proxy[groups == "B"])

        assert mock_validate_sample_sizes.call_count == 2
        np.testing.assert_array_equal(mock_validate_sample_sizes.call_args_list[0][0][0], labeled_mask)
        assert mock_validate_sample_sizes.call_args_list[0][0][1] == "A"
        np.testing.assert_array_equal(mock_validate_sample_sizes.call_args_list[1][0][0], labeled_mask)
        assert mock_validate_sample_sizes.call_args_list[1][0][1] == "B"


def test_preprocess_valid_output(engine, y_true, y_proxy, groups):
    dataset = engine.preprocess(y_true, y_proxy, groups)

    assert set(dataset.keys()) == {"A", "B"}
    for stratum_id in ("A", "B"):
        y_true_labeled, y_proxy_labeled, y_proxy_unlabeled = dataset[stratum_id]
        np.testing.assert_array_equal(y_true_labeled, np.array([5.0, 6.0]))
        np.testing.assert_array_equal(y_proxy_labeled, np.array([4.9, 6.1]))
        np.testing.assert_array_equal(y_proxy_unlabeled, np.array([5.2, 6.1]))


def test_preprocess_allows_constant_proxy_within_stratum(engine):
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([5.0, 5.0, 5.0, 5.0])
    groups = np.array(["A", "A", "A", "A"])

    dataset = engine.preprocess(y_true, y_proxy, groups)

    y_true_labeled, y_proxy_labeled, y_proxy_unlabeled = dataset["A"]
    np.testing.assert_array_equal(y_true_labeled, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(y_proxy_labeled, np.array([5.0, 5.0]))
    np.testing.assert_array_equal(y_proxy_unlabeled, np.array([5.0, 5.0]))


# --- fit_tuning_parameter ---


def test_fit_tuning_parameter_power_tuning_true(engine, dataset):
    tuning_parameter = engine.fit_tuning_parameter(dataset, power_tuning=True)
    assert tuning_parameter["A"] == pytest.approx(0.7843137254901965)
    assert tuning_parameter["B"] == pytest.approx(0.7843137254901965)


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
    tuning_parameter = {"A": 0.7843137254901965, "B": 0.7843137254901965}
    mean, std = engine.compute_mean_and_std(dataset, tuning_parameter)
    assert mean == pytest.approx(5.61764705882353)
    assert std == pytest.approx(0.25043215244009415)


def test_compute_mean_and_std_falls_back_to_unpowered_ppi_for_missing_stratum(engine, dataset):
    tuning_parameter = {"A": 0.5}

    mean, std = engine.compute_mean_and_std(dataset, tuning_parameter)

    assert mean == pytest.approx(5.612500000000001)
    assert std == pytest.approx(0.27528394431931535)
