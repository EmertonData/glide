import numpy as np
import pytest

from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.mean_monitoring_results import PredictionPoweredMeanMonitoringResult
from glide.monitors import EmpiricalPPRM


@pytest.fixture
def y_true():
    return np.array([0.49, 0.51, np.nan, np.nan, 0.5, 0.54, np.nan, np.nan])


@pytest.fixture
def y_proxy():
    return np.array([0.5, 0.5, 0.49, 0.55, 0.52, 0.48, 0.5, 0.52])


@pytest.fixture
def batches():
    return np.array([0, 0, 0, 0, 1, 1, 1, 1])


@pytest.fixture
def monitor():
    return EmpiricalPPRM()


# --- detect ---


def test_detect_is_valid_monitoring_result(monitor, y_true, y_proxy, batches):
    result = monitor.detect(y_true, y_proxy, batches, higher_is_better=False, threshold=0.5)

    assert isinstance(result, PredictionPoweredMeanMonitoringResult)
    assert isinstance(result.confidence_sequence, EmpiricalBernsteinConfidenceSequence)
    assert result.monitor_name == "EmpiricalPPRM"
    assert np.isfinite(result.running_means).all()
    assert (result.running_means >= result.confidence_bounds).all()


def test_detect_metadata(monitor, y_true, y_proxy, batches):
    result = monitor.detect(
        y_true, y_proxy, batches, higher_is_better=True, threshold=0.5, metric_name="accuracy", confidence_level=0.85
    )

    assert result.metric_name == "accuracy"
    assert result.monitor_name == "EmpiricalPPRM"
    assert result.higher_is_better is True
    assert result.alarm_threshold == 0.5
    assert result.confidence_level == 0.85
    np.testing.assert_array_equal(result.batch_n_true, np.array([2, 2]))
    np.testing.assert_array_equal(result.batch_n_proxy, np.array([4, 4]))


def test_detect_custom_confidence_level(monitor, y_true, y_proxy, batches):
    expected_running_means = np.array([0.52, 0.52])
    expected_confidence_bounds = np.array([0.312, 0.416])

    result = monitor.detect(
        y_true, y_proxy, batches, higher_is_better=False, threshold=0.5, metric_name="risk", confidence_level=0.10
    )

    assert result.confidence_level == 0.10
    np.testing.assert_allclose(result.running_means, expected_running_means, atol=0.001)
    np.testing.assert_allclose(result.confidence_bounds, expected_confidence_bounds, atol=0.001)
