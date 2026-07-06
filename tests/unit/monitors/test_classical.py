from unittest.mock import patch

import numpy as np
import pytest

import glide.monitors.classical as classical_module
from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult
from glide.monitors import ClassicalMeanMonitor


@pytest.fixture
def y():
    return np.array([0.1, 0.3, 0.2, 0.4])


@pytest.fixture
def batches():
    return np.array([0, 0, 1, 1])


@pytest.fixture
def monitor():
    return ClassicalMeanMonitor()


# --- _preprocess ---


def test_preprocess_delegates_to_validation(monitor, y, batches):
    with (
        patch.object(classical_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(classical_module, "_validate_non_empty") as mock_validate_non_empty,
        patch.object(classical_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(classical_module, "_validate_bounds") as mock_validate_bounds,
        patch.object(classical_module, "_unique_ordered_batches") as mock_unique_ordered_batches,
    ):
        mock_unique_ordered_batches.return_value = (np.array([0, 1]), np.array([0, 0, 1, 1]))
        monitor._preprocess(
            y,
            batches,
            higher_is_better=False,
            threshold=0.5,
            confidence_level=0.8,
            metric_lower_bound=0.0,
            metric_upper_bound=1.0,
        )

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], batches)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y", "batches"]}

        mock_validate_non_empty.assert_called_once()
        np.testing.assert_array_equal(mock_validate_non_empty.call_args[0][0], y)
        assert mock_validate_non_empty.call_args[0][1] == "y"

        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], batches)
        assert mock_validate_has_no_nan.call_args[0][1] == "batches"

        assert mock_validate_bounds.call_count == 5

        assert mock_validate_bounds.call_args_list[0][0] == (0.8, "confidence_level")
        assert mock_validate_bounds.call_args_list[0][1] == {
            "lower": 0,
            "upper": 1,
            "left_inclusive": False,
            "right_inclusive": False,
        }

        assert mock_validate_bounds.call_args_list[1][0] == (0.0, "metric_lower_bound")
        assert mock_validate_bounds.call_args_list[1][1]["upper"] == 1.0
        assert mock_validate_bounds.call_args_list[1][1]["right_inclusive"] is False
        assert (
            "'metric_lower_bound' must be strictly smaller than 'metric_upper_bound'"
            in mock_validate_bounds.call_args_list[1][1]["error_message"]
        )

        assert mock_validate_bounds.call_args_list[2][0] == (0.5, "threshold")
        assert mock_validate_bounds.call_args_list[2][1]["lower"] == 0.0
        assert mock_validate_bounds.call_args_list[2][1]["upper"] == 1.0
        assert "'threshold' must lie between" in mock_validate_bounds.call_args_list[2][1]["error_message"]

        np.testing.assert_array_equal(mock_validate_bounds.call_args_list[3][0][0], y)
        assert mock_validate_bounds.call_args_list[3][0][1] == "y"
        assert mock_validate_bounds.call_args_list[3][1]["lower"] == 0.0
        assert mock_validate_bounds.call_args_list[3][1]["upper"] == 1.0
        assert "'y' values must lie between" in mock_validate_bounds.call_args_list[3][1]["error_message"]

        assert mock_validate_bounds.call_args_list[4][0][0] == 2
        assert mock_validate_bounds.call_args_list[4][0][1] == "y"
        assert mock_validate_bounds.call_args_list[4][1]["lower"] == 2
        assert (
            "'y' must have at least 2 non-NaN values per batch"
            in mock_validate_bounds.call_args_list[4][1]["error_message"]
        )

        mock_unique_ordered_batches.assert_called_once()
        np.testing.assert_array_equal(mock_unique_ordered_batches.call_args[0][0], batches)


def test_preprocess_known_output(monitor, y, batches):
    risk_y, risk_threshold, batch_codes, batch_n = monitor._preprocess(
        y,
        batches,
        higher_is_better=False,
        threshold=0.5,
        confidence_level=0.8,
        metric_lower_bound=0.0,
        metric_upper_bound=1.0,
    )

    np.testing.assert_allclose(risk_y, y)
    assert risk_threshold == pytest.approx(0.5)
    np.testing.assert_array_equal(batch_codes, np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(batch_n, np.array([2, 2]))


# --- _postprocess ---


def test_postprocess_delegates_to_scaling(monitor):
    risk_running_means = np.array([0.2, 0.25])
    risk_confidence_bounds = np.array([0.1, 0.2])
    risk_batch_mean_estimates = np.array([0.2, 0.3])

    with patch.object(classical_module, "_scale_from_unit_risk") as mock_scale_from_unit_risk:
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


# --- detect ---


def test_detect_is_valid_monitoring_result(monitor, y, batches):
    result = monitor.detect(
        y,
        batches,
        higher_is_better=False,
        threshold=0.5,
    )

    assert isinstance(result, ClassicalMeanMonitoringResult)
    assert isinstance(result.confidence_sequence, EmpiricalBernsteinConfidenceSequence)
    assert result.monitor_name == "ClassicalMeanMonitor"
    assert np.isfinite(result.running_means).all()
    assert (result.running_means >= result.confidence_bounds).all()


def test_detect_metadata(monitor, y, batches):
    result = monitor.detect(
        y, batches, higher_is_better=True, threshold=0.5, metric_name="accuracy", confidence_level=0.85
    )

    assert result.metric_name == "accuracy"
    assert result.monitor_name == "ClassicalMeanMonitor"
    assert result.higher_is_better is True
    assert result.alarm_threshold == 0.5
    assert result.confidence_level == 0.85
    np.testing.assert_array_equal(result.batch_n, np.array([2, 2]))


def test_detect_custom_confidence_level(monitor, y, batches):
    result = monitor.detect(
        y, batches, higher_is_better=False, threshold=0.5, metric_name="perf", confidence_level=0.90
    )

    expected_running_means = np.array([0.2, 0.25])
    expected_confidence_bounds = np.array([-3.551, -1.633])

    assert result.confidence_level == 0.90
    np.testing.assert_allclose(result.running_means, expected_running_means, atol=0.001)
    np.testing.assert_allclose(result.confidence_bounds, expected_confidence_bounds, atol=0.001)
