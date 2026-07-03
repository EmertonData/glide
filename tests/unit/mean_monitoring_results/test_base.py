from unittest.mock import patch

import numpy as np
import pytest

import glide.mean_monitoring_results.base as base_module
from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.mean_monitoring_results import MeanMonitoringResult

# --- MeanMonitoringResult (common attributes and properties) ---


@pytest.fixture
def sequence():
    return EmpiricalBernsteinConfidenceSequence(
        running_mean_estimates=np.array([0.4, 0.6]),
        confidence_bounds=np.array([0.1, 0.55]),
    )


@pytest.fixture
def result_with_alarm(sequence):
    return MeanMonitoringResult(
        metric_name="accuracy",
        monitor_name="Test",
        higher_is_better=True,
        alarm_threshold=0.5,
        confidence_level=0.95,
        batch_mean_estimates=np.array([0.4, 0.8]),
        confidence_sequence=sequence,
    )


@pytest.fixture
def result_without_alarm(sequence):
    return MeanMonitoringResult(
        metric_name="accuracy",
        monitor_name="Test",
        higher_is_better=False,
        alarm_threshold=2.0,
        confidence_level=0.95,
        batch_mean_estimates=np.array([0.4, 0.8]),
        confidence_sequence=sequence,
    )


def test_base_properties(result_with_alarm, result_without_alarm):
    np.testing.assert_array_equal(result_with_alarm.running_means, np.array([0.4, 0.6]))
    np.testing.assert_array_equal(result_with_alarm.confidence_bounds, np.array([0.1, 0.55]))
    assert result_with_alarm.n_batches == 2
    np.testing.assert_array_equal(result_with_alarm.alarms, np.array([True, False]))
    assert result_with_alarm.drift_detected is True
    assert result_with_alarm.first_alarm_index == 0

    np.testing.assert_array_equal(result_without_alarm.alarms, np.array([False, False]))
    assert result_without_alarm.drift_detected is False
    assert result_without_alarm.first_alarm_index is None


def test_base_repr_equals_str_equals_summary(result_with_alarm):
    assert repr(result_with_alarm) == str(result_with_alarm)
    assert str(result_with_alarm) == result_with_alarm.summary()


# --- MeanMonitoringResult.__post_init__ ---


def test_base_post_init_delegates_to_validation(sequence):
    batch_mean_estimates = np.array([0.4, 0.8])
    with (
        patch.object(base_module, "_validate_non_empty") as mock_validate_non_empty,
        patch.object(base_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
    ):
        MeanMonitoringResult(
            metric_name="accuracy",
            monitor_name="Test",
            higher_is_better=True,
            alarm_threshold=0.5,
            confidence_level=0.95,
            batch_mean_estimates=batch_mean_estimates,
            confidence_sequence=sequence,
        )
        mock_validate_non_empty.assert_called_once()
        np.testing.assert_array_equal(mock_validate_non_empty.call_args[0][0], batch_mean_estimates)
        assert mock_validate_non_empty.call_args[0][1] == "batch_mean_estimates"
        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], batch_mean_estimates)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], sequence.running_mean_estimates)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][2], sequence.confidence_bounds)
        assert mock_validate_equal_lengths.call_args[1] == {
            "names": ["batch_mean_estimates", "running_mean_estimates", "confidence_bounds"]
        }


# --- MeanMonitoringResult.__str__ ---


def test_base_str(result_with_alarm):
    expected = (
        "Metric: accuracy\n"
        "Monitor: Test\n"
        "Number of Batches: 2\n"
        "Drift Detected: True\n"
        "First Alarm Index: 0\n"
        "Alarm Threshold: 0.500\n"
        "Running Mean: 0.600\n"
        "Confidence Bound: 0.550\n"
        "Confidence Level: 0.95"
    )
    assert str(result_with_alarm) == expected
