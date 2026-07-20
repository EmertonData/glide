from unittest.mock import patch

import numpy as np
import pytest

import glide.monitors.asymptotic_classical as asymptotic_classical_module
from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult
from glide.monitors import AsymptoticClassicalMeanMonitor


@pytest.fixture
def y():
    return np.array([0.49, 0.51, 0.5, 0.54])


@pytest.fixture
def batches():
    return np.array([0, 0, 1, 1])


@pytest.fixture
def monitor():
    return AsymptoticClassicalMeanMonitor()


# --- detect ---


def test_detect_delegates_to_validation(monitor, y, batches):
    with patch.object(asymptotic_classical_module, "_validate_bounds") as mock_validate_bounds:
        monitor.detect(y, batches, higher_is_better=False, threshold=0.5)

        mock_validate_bounds.assert_called_once_with(
            0.8,
            "confidence_level",
            lower=0.5,
            upper=1,
            left_inclusive=False,
            right_inclusive=False,
            error_message="'confidence_level' must be in (0.5, 1) for the asymptotic monitor; got 0.8.",
        )


def test_detect_is_valid_monitoring_result(monitor, y, batches):
    result = monitor.detect(
        y,
        batches,
        higher_is_better=False,
        threshold=0.5,
    )

    assert isinstance(result, ClassicalMeanMonitoringResult)
    assert isinstance(result.confidence_sequence, AsymptoticConfidenceSequence)
    assert result.monitor_name == "AsymptoticClassicalMeanMonitor"
    assert np.isfinite(result.running_means).all()
    assert (result.running_means >= result.confidence_bounds).all()


def test_detect_metadata(monitor, y, batches):
    result = monitor.detect(
        y, batches, higher_is_better=True, threshold=0.5, metric_name="accuracy", confidence_level=0.85
    )

    assert result.metric_name == "accuracy"
    assert result.monitor_name == "AsymptoticClassicalMeanMonitor"
    assert result.higher_is_better is True
    assert result.alarm_threshold == 0.5
    assert result.confidence_level == 0.85
    np.testing.assert_array_equal(result.batch_n, np.array([2, 2]))


def test_detect_custom_confidence_level(monitor, y, batches):
    expected_running_means = np.array([0.5, 0.51])
    expected_confidence_bounds = np.array([0.472, 0.484])

    result = monitor.detect(
        y, batches, higher_is_better=False, threshold=0.5, metric_name="perf", confidence_level=0.85
    )

    assert result.confidence_level == 0.85
    np.testing.assert_allclose(result.running_means, expected_running_means, atol=0.001)
    np.testing.assert_allclose(result.confidence_bounds, expected_confidence_bounds, atol=0.001)


def test_detect_invalid_confidence_level(monitor, y, batches):
    with pytest.raises(ValueError, match=r"'confidence_level' must be in \(0.5, 1\) for the asymptotic monitor"):
        monitor.detect(y, batches, higher_is_better=False, threshold=0.5, confidence_level=0.5)
