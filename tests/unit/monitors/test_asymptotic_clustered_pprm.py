from unittest.mock import patch

import numpy as np
import pytest

import glide.monitors.asymptotic_clustered_pprm as asymptotic_clustered_pprm_module
from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.mean_monitoring_results import PredictionPoweredMeanMonitoringResult
from glide.monitors import AsymptoticClusteredPPRM


@pytest.fixture
def y_true():
    return np.array(
        [0.4, 0.6, 0.5, 0.7, np.nan, np.nan, np.nan, np.nan, 0.3, 0.5, 0.6, 0.8, np.nan, np.nan, np.nan, np.nan]
    )


@pytest.fixture
def y_proxy():
    return np.array([0.5, 0.5, 0.6, 0.6, 0.4, 0.7, 0.5, 0.5, 0.4, 0.4, 0.7, 0.7, 0.5, 0.6, 0.3, 0.5])


@pytest.fixture
def clusters():
    return np.array(["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F", "F", "G", "G", "H", "H"])


@pytest.fixture
def batches():
    return np.repeat(np.array([0, 1]), 8)


@pytest.fixture
def monitor():
    return AsymptoticClusteredPPRM()


# --- detect ---


def test_detect_is_valid_monitoring_result(monitor, y_true, y_proxy, clusters, batches):
    result = monitor.detect(y_true, y_proxy, clusters, batches, higher_is_better=False, threshold=0.5)

    assert isinstance(result, PredictionPoweredMeanMonitoringResult)
    assert isinstance(result.confidence_sequence, AsymptoticConfidenceSequence)
    assert result.monitor_name == "AsymptoticClusteredPPRM"
    assert np.isfinite(result.running_means).all()
    assert (result.running_means >= result.confidence_bounds).all()


def test_detect_metadata(monitor, y_true, y_proxy, clusters, batches):
    result = monitor.detect(
        y_true,
        y_proxy,
        clusters,
        batches,
        higher_is_better=True,
        threshold=0.5,
        metric_name="accuracy",
        confidence_level=0.85,
    )

    assert result.metric_name == "accuracy"
    assert result.monitor_name == "AsymptoticClusteredPPRM"
    assert result.higher_is_better is True
    assert result.alarm_threshold == 0.5
    assert result.confidence_level == 0.85
    np.testing.assert_array_equal(result.batch_n_true, np.array([4, 4]))
    np.testing.assert_array_equal(result.batch_n_proxy, np.array([8, 8]))


def test_detect_custom_confidence_level(monitor, y_true, y_proxy, clusters, batches):
    expected_running_means = np.array([0.525, 0.497])
    expected_confidence_bounds = np.array([0.433, 0.396])

    result = monitor.detect(
        y_true,
        y_proxy,
        clusters,
        batches,
        higher_is_better=False,
        threshold=0.5,
        metric_name="risk",
        confidence_level=0.85,
    )

    assert result.confidence_level == 0.85
    np.testing.assert_allclose(result.running_means, expected_running_means, atol=0.001)
    np.testing.assert_allclose(result.confidence_bounds, expected_confidence_bounds, atol=0.001)


def test_detect_delegates_to_validation(monitor, y_true, y_proxy, clusters, batches):
    with patch.object(
        asymptotic_clustered_pprm_module, "_validate_clusters_unique_across_batches"
    ) as mock_validate_clusters_unique_across_batches:
        monitor.detect(y_true, y_proxy, clusters, batches, higher_is_better=False, threshold=0.5)

        mock_validate_clusters_unique_across_batches.assert_called_once()
        np.testing.assert_array_equal(mock_validate_clusters_unique_across_batches.call_args[0][0], clusters)
        np.testing.assert_array_equal(mock_validate_clusters_unique_across_batches.call_args[0][1], batches)
