import numpy as np
import pytest

from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult
from glide.monitors import AsymptoticStratifiedClassicalRM


@pytest.fixture
def y():
    return np.array(
        [0.49, 0.51, np.nan, np.nan, 0.4, 0.6, np.nan, 0.5, 0.5, 0.54, np.nan, np.nan, 0.45, 0.55, 0.5, np.nan]
    )


@pytest.fixture
def groups():
    return np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])


@pytest.fixture
def batches():
    return np.repeat(np.array([0, 1]), 8)


@pytest.fixture
def monitor():
    return AsymptoticStratifiedClassicalRM()


# --- detect ---


def test_detect_is_valid_monitoring_result(monitor, y, groups, batches):
    result = monitor.detect(y, groups, batches, higher_is_better=False, threshold=0.5)

    assert isinstance(result, ClassicalMeanMonitoringResult)
    assert isinstance(result.confidence_sequence, AsymptoticConfidenceSequence)
    assert result.monitor_name == "AsymptoticStratifiedClassicalRM"
    assert np.isfinite(result.running_means).all()
    assert (result.running_means >= result.confidence_bounds).all()


def test_detect_metadata(monitor, y, groups, batches):
    result = monitor.detect(
        y,
        groups,
        batches,
        higher_is_better=True,
        threshold=0.5,
        metric_name="accuracy",
        confidence_level=0.85,
    )

    assert result.metric_name == "accuracy"
    assert result.monitor_name == "AsymptoticStratifiedClassicalRM"
    assert result.higher_is_better is True
    assert result.alarm_threshold == 0.5
    assert result.confidence_level == 0.85
    np.testing.assert_array_equal(result.batch_n, np.array([5, 5]))


def test_detect_custom_confidence_level(monitor, y, groups, batches):
    expected_batch_mean_estimates = np.array([0.5, 0.51])
    expected_running_means = np.array([0.5, 0.505])
    expected_confidence_bounds = np.array([0.4313, 0.4655])

    result = monitor.detect(
        y, groups, batches, higher_is_better=False, threshold=0.5, metric_name="risk", confidence_level=0.85
    )

    assert result.confidence_level == 0.85
    np.testing.assert_allclose(result.batch_mean_estimates, expected_batch_mean_estimates, atol=0.0001)
    np.testing.assert_allclose(result.running_means, expected_running_means, atol=0.0001)
    np.testing.assert_allclose(result.confidence_bounds, expected_confidence_bounds, atol=0.0001)
