from unittest.mock import patch

import numpy as np
import pytest

import glide.monitors.asymptotic_clustered_classical as asymptotic_clustered_classical_module
from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult
from glide.monitors import AsymptoticClusteredClassicalRM


@pytest.fixture
def y():
    return np.array([0.4, 0.6, 0.5, 0.7, 0.2, 0.4, np.nan, np.nan, 0.3, 0.5, 0.6, 0.8, 0.1, 0.3, np.nan, np.nan])


@pytest.fixture
def clusters():
    return np.array(["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F", "F", "G", "G", "H", "H"])


@pytest.fixture
def batches():
    return np.repeat(np.array([0, 1]), 8)


@pytest.fixture
def monitor():
    return AsymptoticClusteredClassicalRM()


# --- detect ---


def test_detect_delegates_to_validation(monitor, y, clusters, batches):
    with patch.object(
        asymptotic_clustered_classical_module, "_validate_clusters_unique_across_batches"
    ) as mock_validate_clusters_unique_across_batches:
        monitor.detect(y, clusters, batches, higher_is_better=False, threshold=0.5)

        mock_validate_clusters_unique_across_batches.assert_called_once()
        np.testing.assert_array_equal(mock_validate_clusters_unique_across_batches.call_args[0][0], clusters)
        np.testing.assert_array_equal(mock_validate_clusters_unique_across_batches.call_args[0][1], batches)


def test_detect_is_valid_monitoring_result(monitor, y, clusters, batches):
    result = monitor.detect(y, clusters, batches, higher_is_better=False, threshold=0.5)

    assert isinstance(result, ClassicalMeanMonitoringResult)
    assert isinstance(result.confidence_sequence, AsymptoticConfidenceSequence)
    assert result.monitor_name == "AsymptoticClusteredClassicalRM"
    assert np.isfinite(result.running_means).all()
    assert (result.running_means >= result.confidence_bounds).all()


def test_detect_metadata(monitor, y, clusters, batches):
    result = monitor.detect(
        y,
        clusters,
        batches,
        higher_is_better=True,
        threshold=0.5,
        metric_name="accuracy",
        confidence_level=0.85,
    )

    assert result.metric_name == "accuracy"
    assert result.monitor_name == "AsymptoticClusteredClassicalRM"
    assert result.higher_is_better is True
    assert result.alarm_threshold == 0.5
    assert result.confidence_level == 0.85
    np.testing.assert_array_equal(result.batch_n, np.array([6, 6]))


def test_detect_custom_confidence_level(monitor, y, clusters, batches):
    expected_batch_mean_estimates = np.array([0.4667, 0.4333])
    expected_running_means = np.array([0.4667, 0.45])
    expected_confidence_bounds = np.array([0.2328, 0.2533])

    result = monitor.detect(
        y, clusters, batches, higher_is_better=False, threshold=0.5, metric_name="risk", confidence_level=0.85
    )

    assert result.confidence_level == 0.85
    np.testing.assert_allclose(result.batch_mean_estimates, expected_batch_mean_estimates, atol=0.0001)
    np.testing.assert_allclose(result.running_means, expected_running_means, atol=0.0001)
    np.testing.assert_allclose(result.confidence_bounds, expected_confidence_bounds, atol=0.0001)
