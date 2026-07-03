import numpy as np
import pytest

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
def result_higher_is_better(sequence):
    return MeanMonitoringResult(
        metric_name="accuracy",
        monitor_name="Test",
        higher_is_better=True,
        alarm_threshold=0.5,
        confidence_level=0.95,
        batch_identifiers=np.array([0, 1]),
        batch_mean_estimates=np.array([0.4, 0.8]),
        confidence_sequence=sequence,
    )


@pytest.fixture
def result_lower_is_better(sequence):
    return MeanMonitoringResult(
        metric_name="loss",
        monitor_name="Test",
        higher_is_better=False,
        alarm_threshold=0.5,
        confidence_level=0.95,
        batch_identifiers=np.array([0, 1]),
        batch_mean_estimates=np.array([0.4, 0.8]),
        confidence_sequence=sequence,
    )


@pytest.fixture
def result_no_alarm(sequence):
    return MeanMonitoringResult(
        metric_name="accuracy",
        monitor_name="Test",
        higher_is_better=True,
        alarm_threshold=-1.0,
        confidence_level=0.95,
        batch_identifiers=np.array([0, 1]),
        batch_mean_estimates=np.array([0.4, 0.8]),
        confidence_sequence=sequence,
    )


def test_base_running_means(result_higher_is_better):
    np.testing.assert_array_equal(result_higher_is_better.running_means, np.array([0.4, 0.6]))


def test_base_confidence_bounds(result_higher_is_better):
    np.testing.assert_array_equal(result_higher_is_better.confidence_bounds, np.array([0.1, 0.55]))


def test_base_alarms_higher_is_better(result_higher_is_better):
    np.testing.assert_array_equal(result_higher_is_better.alarms, np.array([True, False]))


def test_base_alarms_lower_is_better(result_lower_is_better):
    np.testing.assert_array_equal(result_lower_is_better.alarms, np.array([False, True]))


def test_base_n_batches(result_higher_is_better):
    assert result_higher_is_better.n_batches == 2


def test_base_drift_detected_true(result_higher_is_better):
    assert result_higher_is_better.drift_detected is True


def test_base_drift_detected_false(result_no_alarm):
    assert result_no_alarm.drift_detected is False


def test_base_first_alarm_index_at_zero(result_higher_is_better):
    assert result_higher_is_better.first_alarm_index == 0


def test_base_first_alarm_index_not_at_zero(result_lower_is_better):
    assert result_lower_is_better.first_alarm_index == 1


def test_base_first_alarm_index_none(result_no_alarm):
    assert result_no_alarm.first_alarm_index is None


def test_base_repr_equals_str_equals_summary(result_higher_is_better):
    assert repr(result_higher_is_better) == str(result_higher_is_better)
    assert str(result_higher_is_better) == result_higher_is_better.summary()


# --- MeanMonitoringResult.__str__ ---


def test_base_str(result_higher_is_better):
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
    assert str(result_higher_is_better) == expected
