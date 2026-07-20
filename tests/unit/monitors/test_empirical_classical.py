import numpy as np
import pytest

from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult
from glide.monitors import EmpiricalClassicalMeanMonitor


@pytest.fixture
def y():
    return np.array([0.49, 0.51, 0.5, 0.54])


@pytest.fixture
def batches():
    return np.array([0, 0, 1, 1])


@pytest.fixture
def monitor():
    return EmpiricalClassicalMeanMonitor()


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
    assert result.monitor_name == "EmpiricalClassicalMeanMonitor"
    assert np.isfinite(result.running_means).all()
    assert (result.running_means >= result.confidence_bounds).all()


def test_detect_metadata(monitor, y, batches):
    result = monitor.detect(
        y, batches, higher_is_better=True, threshold=0.5, metric_name="accuracy", confidence_level=0.85
    )

    assert result.metric_name == "accuracy"
    assert result.monitor_name == "EmpiricalClassicalMeanMonitor"
    assert result.higher_is_better is True
    assert result.alarm_threshold == 0.5
    assert result.confidence_level == 0.85
    np.testing.assert_array_equal(result.batch_n, np.array([2, 2]))


def test_detect_custom_confidence_level(monitor, y, batches):
    expected_running_means = np.array([0.5, 0.51])
    expected_confidence_bounds = np.array([0.292, 0.406])

    result = monitor.detect(
        y, batches, higher_is_better=False, threshold=0.5, metric_name="perf", confidence_level=0.10
    )

    assert result.confidence_level == 0.10
    np.testing.assert_allclose(result.running_means, expected_running_means, atol=0.001)
    np.testing.assert_allclose(result.confidence_bounds, expected_confidence_bounds, atol=0.001)
