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
        patch.object(classical_module, "_validate_min_samples") as mock_validate_min_samples,
        patch.object(classical_module, "_validate_bounds") as mock_validate_bounds,
        patch.object(classical_module, "_unique_ordered_batches") as mock_unique_ordered_batches,
    ):
        mock_unique_ordered_batches.return_value = (np.array([0, 1]), np.array([0, 0, 1, 1]))
        monitor._preprocess(
            y, batches, higher_is_better=False, threshold=0.5, metric_lower_bound=0.0, metric_upper_bound=1.0
        )

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], batches)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y", "batches"]}

        mock_validate_min_samples.assert_called_once()
        np.testing.assert_array_equal(mock_validate_min_samples.call_args[0][0], y)
        assert mock_validate_min_samples.call_args[0][1] == "y"

        assert mock_validate_bounds.call_count == 2
        np.testing.assert_array_equal(mock_validate_bounds.call_args_list[0][0][0], y)
        assert mock_validate_bounds.call_args_list[0][0][1] == "y"
        assert mock_validate_bounds.call_args_list[0][1]["lower"] == 0.0
        assert mock_validate_bounds.call_args_list[0][1]["upper"] == 1.0
        assert "'y' values must lie between" in mock_validate_bounds.call_args_list[0][1]["error_message"]

        assert mock_validate_bounds.call_args_list[1][0][0] == 2
        assert mock_validate_bounds.call_args_list[1][0][1] == "y"
        assert mock_validate_bounds.call_args_list[1][1]["lower"] == 2
        assert (
            "'y' must have at least 2 non-NaN values per batch"
            in mock_validate_bounds.call_args_list[1][1]["error_message"]
        )

        mock_unique_ordered_batches.assert_called_once()
        np.testing.assert_array_equal(mock_unique_ordered_batches.call_args[0][0], batches)


def test_preprocess_valid_output(monitor, y, batches):
    risk_y, risk_threshold, batch_codes, batch_n = monitor._preprocess(
        y, batches, higher_is_better=False, threshold=0.5, metric_lower_bound=0.0, metric_upper_bound=1.0
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


def test_detect_returns_classical_mean_monitoring_result(monitor, y, batches):
    result = monitor.detect(
        y, batches, higher_is_better=False, threshold=0.5, metric_lower_bound=0.0, metric_upper_bound=1.0
    )
    assert isinstance(result, ClassicalMeanMonitoringResult)


def test_detect_confidence_sequence_is_empirical_bernstein(monitor, y, batches):
    result = monitor.detect(
        y, batches, higher_is_better=False, threshold=0.5, metric_lower_bound=0.0, metric_upper_bound=1.0
    )

    assert isinstance(result.confidence_sequence, EmpiricalBernsteinConfidenceSequence)
    np.testing.assert_array_equal(result.running_means, result.confidence_sequence.running_mean_estimates)
    np.testing.assert_array_equal(result.confidence_bounds, result.confidence_sequence.confidence_bounds)
    np.testing.assert_array_equal(result.alarms, result.confidence_sequence.test_null_hypothesis(0.5, "larger"))


def test_detect_batch_mean_estimates(monitor, y, batches):
    result = monitor.detect(y, batches, higher_is_better=False, threshold=0.5, confidence_level=0.8)

    np.testing.assert_allclose(result.batch_mean_estimates, np.array([0.2, 0.3]))
    np.testing.assert_allclose(result.running_means, np.array([0.2, 0.25]))
    np.testing.assert_array_equal(result.batch_n, np.array([2, 2]))
    assert result.drift_detected is False


def test_detect_drops_nan_per_batch(monitor):
    y_true = np.array([0.1, np.nan, 0.3, 0.2, np.nan, 0.4])
    batches = np.array([0, 0, 0, 1, 1, 1])

    result = monitor.detect(y_true, batches, higher_is_better=False, threshold=0.5)

    np.testing.assert_allclose(result.batch_mean_estimates, np.array([0.2, 0.3]))
    np.testing.assert_array_equal(result.batch_n, np.array([2, 2]))


def test_detect_delegates_to_validation(monitor, y, batches):
    mock_preprocess_return = (y, 0.5, np.array([0, 0, 1, 1]), np.array([2, 2]))
    with (
        patch.object(classical_module, "_validate_bounds") as mock_validate_bounds,
        patch.object(monitor, "_preprocess", return_value=mock_preprocess_return),
    ):
        monitor.detect(
            y,
            batches,
            higher_is_better=False,
            threshold=0.5,
            confidence_level=0.8,
            metric_lower_bound=0.0,
            metric_upper_bound=1.0,
        )

        assert mock_validate_bounds.call_count == 3

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


def test_detect_metadata(monitor, y, batches):
    result = monitor.detect(
        y, batches, higher_is_better=True, threshold=0.5, metric_name="accuracy", confidence_level=0.9
    )

    assert result.metric_name == "accuracy"
    assert result.monitor_name == "ClassicalMeanMonitor"
    assert result.higher_is_better is True
    assert result.alarm_threshold == 0.5
    assert result.confidence_level == 0.9
