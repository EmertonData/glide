from unittest.mock import patch

import numpy as np
import pytest

import glide.monitors.ppi as ppi_module
from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.mean_monitoring_results import PredictionPoweredMeanMonitoringResult
from glide.monitors import PPIMeanMonitor


@pytest.fixture
def y_true():
    return np.array([0.1, 0.3, np.nan, np.nan, 0.2, 0.4, np.nan, np.nan])


@pytest.fixture
def y_proxy():
    return np.array([0.1, 0.3, 0.15, 0.35, 0.2, 0.4, 0.25, 0.45])


@pytest.fixture
def batches():
    return np.array([0, 0, 0, 0, 1, 1, 1, 1])


@pytest.fixture
def monitor():
    return PPIMeanMonitor()


# --- _preprocess ---


def test_preprocess_delegates_to_validation(monitor, y_true, y_proxy, batches):
    with (
        patch.object(ppi_module, "_validate_non_empty") as mock_validate_non_empty,
        patch.object(ppi_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(ppi_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(ppi_module, "_validate_bounds") as mock_validate_bounds,
        patch.object(ppi_module, "_validate_y_proxy") as mock_validate_y_proxy,
        patch.object(ppi_module, "_validate_y_true") as mock_validate_y_true,
        patch.object(ppi_module, "_unique_ordered_batches") as mock_unique_ordered_batches,
    ):
        mock_unique_ordered_batches.return_value = (np.array([0, 1]), np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        monitor._preprocess(
            y_true,
            y_proxy,
            batches,
            higher_is_better=False,
            threshold=0.5,
            confidence_level=0.8,
            max_tuning_parameter=1.0,
            metric_lower_bound=0.0,
            metric_upper_bound=1.0,
        )

        mock_validate_non_empty.assert_called_once()
        np.testing.assert_array_equal(mock_validate_non_empty.call_args[0][0], y_true)
        assert mock_validate_non_empty.call_args[0][1] == "y_true"

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], y_proxy)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][2], batches)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y_true", "y_proxy", "batches"]}

        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], batches)
        assert mock_validate_has_no_nan.call_args[0][1] == "batches"

        mock_validate_y_proxy.assert_called_once()
        np.testing.assert_array_equal(mock_validate_y_proxy.call_args[0][0], y_proxy)

        mock_validate_y_true.assert_called_once()
        np.testing.assert_array_equal(mock_validate_y_true.call_args[0][0], y_true)

        mock_unique_ordered_batches.assert_called_once()
        np.testing.assert_array_equal(mock_unique_ordered_batches.call_args[0][0], batches)

        assert mock_validate_bounds.call_count == 8

        assert mock_validate_bounds.call_args_list[0][0] == (0.8, "confidence_level")
        assert mock_validate_bounds.call_args_list[0][1] == {
            "lower": 0,
            "upper": 1,
            "left_inclusive": False,
            "right_inclusive": False,
        }

        assert mock_validate_bounds.call_args_list[1][0] == (1.0, "max_tuning_parameter")
        assert mock_validate_bounds.call_args_list[1][1] == {"lower": 0, "left_inclusive": False}

        assert mock_validate_bounds.call_args_list[2][0] == (0.0, "metric_lower_bound")
        assert mock_validate_bounds.call_args_list[2][1]["upper"] == 1.0
        assert mock_validate_bounds.call_args_list[2][1]["right_inclusive"] is False
        assert (
            "'metric_lower_bound' must be strictly smaller than 'metric_upper_bound'"
            in mock_validate_bounds.call_args_list[2][1]["error_message"]
        )

        assert mock_validate_bounds.call_args_list[3][0] == (0.5, "threshold")
        assert mock_validate_bounds.call_args_list[3][1]["lower"] == 0.0
        assert mock_validate_bounds.call_args_list[3][1]["upper"] == 1.0
        assert "'threshold' must lie between" in mock_validate_bounds.call_args_list[3][1]["error_message"]

        np.testing.assert_array_equal(mock_validate_bounds.call_args_list[4][0][0], np.array([0.1, 0.3, 0.2, 0.4]))
        assert mock_validate_bounds.call_args_list[4][0][1] == "y_true"
        assert "'y_true' values must lie between" in mock_validate_bounds.call_args_list[4][1]["error_message"]

        np.testing.assert_array_equal(mock_validate_bounds.call_args_list[5][0][0], y_proxy)
        assert mock_validate_bounds.call_args_list[5][0][1] == "y_proxy"
        assert "'y_proxy' values must lie between" in mock_validate_bounds.call_args_list[5][1]["error_message"]

        assert mock_validate_bounds.call_args_list[6][0] == (2, "y_true")
        assert mock_validate_bounds.call_args_list[6][1]["lower"] == 2
        assert (
            "'y_true' must have at least 2 labeled values per batch"
            in mock_validate_bounds.call_args_list[6][1]["error_message"]
        )

        assert mock_validate_bounds.call_args_list[7][0] == (2, "y_true")
        assert mock_validate_bounds.call_args_list[7][1]["lower"] == 2
        assert (
            "'y_true' must have at least 2 unlabeled values per batch"
            in mock_validate_bounds.call_args_list[7][1]["error_message"]
        )


def test_preprocess_known_output(monitor, y_true, y_proxy, batches):
    risk_y_true, risk_y_proxy, risk_threshold, batch_codes, batch_n_true, batch_n_proxy = monitor._preprocess(
        y_true,
        y_proxy,
        batches,
        higher_is_better=False,
        threshold=0.5,
        confidence_level=0.8,
        max_tuning_parameter=1.0,
        metric_lower_bound=0.0,
        metric_upper_bound=1.0,
    )

    np.testing.assert_allclose(risk_y_true, y_true)
    np.testing.assert_allclose(risk_y_proxy, y_proxy)
    assert risk_threshold == pytest.approx(0.5)
    np.testing.assert_array_equal(batch_codes, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    np.testing.assert_array_equal(batch_n_true, np.array([2, 2]))
    np.testing.assert_array_equal(batch_n_proxy, np.array([4, 4]))


# --- _postprocess ---


def test_postprocess_delegates_to_scaling(monitor):
    risk_running_means = np.array([0.2, 0.25])
    risk_confidence_bounds = np.array([0.1, 0.2])
    risk_batch_mean_estimates = np.array([0.2, 0.3])

    with patch.object(ppi_module, "_scale_from_unit_risk") as mock_scale_from_unit_risk:
        monitor._postprocess(
            risk_running_means,
            risk_confidence_bounds,
            risk_batch_mean_estimates,
            higher_is_better=True,
            metric_lower_bound=0.0,
            metric_upper_bound=1.0,
        )

    assert mock_scale_from_unit_risk.call_count == 3
    np.testing.assert_array_equal(mock_scale_from_unit_risk.call_args_list[0][0][0], risk_running_means)
    np.testing.assert_array_equal(mock_scale_from_unit_risk.call_args_list[1][0][0], risk_confidence_bounds)
    np.testing.assert_array_equal(mock_scale_from_unit_risk.call_args_list[2][0][0], risk_batch_mean_estimates)


def test_postprocess_clips_risk_values(monitor):
    risk_running_means = np.array([-0.5, 1.5])
    risk_confidence_bounds = np.array([-0.5, 1.5])
    risk_batch_mean_estimates = np.array([-0.5, 1.5])

    running_means, confidence_bounds, batch_mean_estimates = monitor._postprocess(
        risk_running_means,
        risk_confidence_bounds,
        risk_batch_mean_estimates,
        higher_is_better=False,
        metric_lower_bound=0.0,
        metric_upper_bound=1.0,
    )

    np.testing.assert_allclose(running_means, np.array([0.0, 1.0]))
    np.testing.assert_allclose(confidence_bounds, np.array([0.0, 1.0]))
    np.testing.assert_allclose(batch_mean_estimates, np.array([0.0, 1.0]))


# --- detect ---


def test_detect_is_valid_monitoring_result(monitor, y_true, y_proxy, batches):
    result = monitor.detect(y_true, y_proxy, batches, higher_is_better=False, threshold=0.5)

    assert isinstance(result, PredictionPoweredMeanMonitoringResult)
    assert isinstance(result.confidence_sequence, EmpiricalBernsteinConfidenceSequence)
    assert result.monitor_name == "PPIMeanMonitor"
    assert np.isfinite(result.running_means).all()
    assert (result.running_means >= result.confidence_bounds).all()


def test_detect_metadata(monitor, y_true, y_proxy, batches):
    result = monitor.detect(
        y_true, y_proxy, batches, higher_is_better=True, threshold=0.5, metric_name="accuracy", confidence_level=0.85
    )

    assert result.metric_name == "accuracy"
    assert result.monitor_name == "PPIMeanMonitor"
    assert result.higher_is_better is True
    assert result.alarm_threshold == 0.5
    assert result.confidence_level == 0.85
    np.testing.assert_array_equal(result.batch_n_true, np.array([2, 2]))
    np.testing.assert_array_equal(result.batch_n_proxy, np.array([4, 4]))


def test_detect_custom_confidence_level(monitor, y_true, y_proxy, batches):
    expected_running_means = np.array([0.25, 0.292647])
    expected_confidence_bounds = np.array([0.0, 0.0])

    result = monitor.detect(
        y_true, y_proxy, batches, higher_is_better=False, threshold=0.5, metric_name="risk", confidence_level=0.90
    )

    assert result.confidence_level == 0.90
    np.testing.assert_allclose(result.running_means, expected_running_means, atol=0.001)
    np.testing.assert_allclose(result.confidence_bounds, expected_confidence_bounds, atol=0.001)
