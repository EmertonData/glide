import numpy as np
import pytest

from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult
from glide.monitors import AsymptoticIPWClassicalRM


@pytest.fixture
def y():
    return np.array([0.2, 0.4, np.nan, np.nan, 0.5, np.nan, 0.7, 0.3])


@pytest.fixture
def sampling_probability():
    return np.array([0.5, 0.8, 0.4, 0.0, 0.5, 0.5, 0.7, 0.6])


@pytest.fixture
def batches():
    return np.repeat(np.arange(2), 4)


@pytest.fixture
def monitor():
    return AsymptoticIPWClassicalRM()


# --- detect ---


def test_detect_is_valid_monitoring_result(monitor, y, sampling_probability, batches):
    result = monitor.detect(y, sampling_probability, batches, higher_is_better=False, threshold=0.5)

    assert isinstance(result, ClassicalMeanMonitoringResult)
    assert isinstance(result.confidence_sequence, AsymptoticConfidenceSequence)
    assert result.monitor_name == "AsymptoticIPWClassicalRM"
    assert np.isfinite(result.running_means).all()
    assert (result.running_means >= result.confidence_bounds).all()


def test_detect_metadata(monitor, y, sampling_probability, batches):
    result = monitor.detect(
        y,
        sampling_probability,
        batches,
        higher_is_better=True,
        threshold=0.5,
        metric_name="accuracy",
        confidence_level=0.85,
    )

    assert result.metric_name == "accuracy"
    assert result.monitor_name == "AsymptoticIPWClassicalRM"
    assert result.higher_is_better is True
    assert result.alarm_threshold == 0.5
    assert result.confidence_level == 0.85
    np.testing.assert_array_equal(result.batch_n, np.array([2, 3]))


def test_detect_custom_confidence_level(monitor, y, sampling_probability, batches):
    expected_batch_mean_estimates = np.array([0.3, 0.625])
    expected_running_means = np.array([0.3, 0.4625])
    expected_confidence_bounds = np.array([-0.0995, 0.1338])

    result = monitor.detect(
        y, sampling_probability, batches, higher_is_better=False, threshold=0.5, confidence_level=0.85
    )

    np.testing.assert_allclose(result.batch_mean_estimates, expected_batch_mean_estimates, atol=1e-9)
    np.testing.assert_allclose(result.running_means, expected_running_means, atol=1e-9)
    np.testing.assert_allclose(result.confidence_bounds, expected_confidence_bounds, atol=0.0001)


def test_detect_higher_is_better_keeps_sampling_probability(monitor, y, sampling_probability, batches):
    expected_confidence_bounds = np.array([0.6995, 0.7912])

    result = monitor.detect(
        y, sampling_probability, batches, higher_is_better=True, threshold=0.5, confidence_level=0.85
    )

    np.testing.assert_allclose(result.batch_mean_estimates, np.array([0.3, 0.625]), atol=1e-9)
    np.testing.assert_allclose(result.confidence_bounds, expected_confidence_bounds, atol=0.0001)
