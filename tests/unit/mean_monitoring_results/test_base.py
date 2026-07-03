import numpy as np

from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.mean_monitoring_results import MeanMonitoringResult

# --- MeanMonitoringResult (common attributes and properties) ---

_SEQUENCE = EmpiricalBernsteinConfidenceSequence(
    running_mean_estimates=np.array([0.4, 0.6]),
    confidence_bounds=np.array([0.1, 0.55]),
)
_RESULT_HIGHER_IS_BETTER = MeanMonitoringResult(
    metric_name="accuracy",
    monitor_name="Test",
    higher_is_better=True,
    alarm_threshold=0.5,
    confidence_level=0.95,
    batch_identifiers=np.array([0, 1]),
    batch_mean_estimates=np.array([0.4, 0.8]),
    confidence_sequence=_SEQUENCE,
)
_RESULT_LOWER_IS_BETTER = MeanMonitoringResult(
    metric_name="loss",
    monitor_name="Test",
    higher_is_better=False,
    alarm_threshold=0.5,
    confidence_level=0.95,
    batch_identifiers=np.array([0, 1]),
    batch_mean_estimates=np.array([0.4, 0.8]),
    confidence_sequence=_SEQUENCE,
)
_RESULT_NO_ALARM = MeanMonitoringResult(
    metric_name="accuracy",
    monitor_name="Test",
    higher_is_better=True,
    alarm_threshold=-1.0,
    confidence_level=0.95,
    batch_identifiers=np.array([0, 1]),
    batch_mean_estimates=np.array([0.4, 0.8]),
    confidence_sequence=_SEQUENCE,
)


def test_base_running_means():
    np.testing.assert_array_equal(_RESULT_HIGHER_IS_BETTER.running_means, np.array([0.4, 0.6]))


def test_base_confidence_bounds():
    np.testing.assert_array_equal(_RESULT_HIGHER_IS_BETTER.confidence_bounds, np.array([0.1, 0.55]))


def test_base_alarms_higher_is_better():
    np.testing.assert_array_equal(_RESULT_HIGHER_IS_BETTER.alarms, np.array([True, False]))


def test_base_alarms_lower_is_better():
    np.testing.assert_array_equal(_RESULT_LOWER_IS_BETTER.alarms, np.array([False, True]))


def test_base_n_batches():
    assert _RESULT_HIGHER_IS_BETTER.n_batches == 2


def test_base_drift_detected_true():
    assert _RESULT_HIGHER_IS_BETTER.drift_detected is True


def test_base_drift_detected_false():
    assert _RESULT_NO_ALARM.drift_detected is False


def test_base_first_alarm_index_at_zero():
    assert _RESULT_HIGHER_IS_BETTER.first_alarm_index == 0


def test_base_first_alarm_index_not_at_zero():
    assert _RESULT_LOWER_IS_BETTER.first_alarm_index == 1


def test_base_first_alarm_index_none():
    assert _RESULT_NO_ALARM.first_alarm_index is None


def test_base_repr_equals_str_equals_summary():
    assert repr(_RESULT_HIGHER_IS_BETTER) == str(_RESULT_HIGHER_IS_BETTER)
    assert str(_RESULT_HIGHER_IS_BETTER) == _RESULT_HIGHER_IS_BETTER.summary()


# --- MeanMonitoringResult.__str__ ---


def test_base_str():
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
    assert str(_RESULT_HIGHER_IS_BETTER) == expected
