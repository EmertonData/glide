from unittest.mock import patch

import numpy as np
import pytest

import glide.mean_monitoring_results.prediction_powered as prediction_powered_module
from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.mean_monitoring_results import PredictionPoweredMeanMonitoringResult

# --- PredictionPoweredMeanMonitoringResult ---


@pytest.fixture
def sequence():
    return EmpiricalBernsteinConfidenceSequence(
        running_mean_estimates=np.array([0.4, 0.6]),
        confidence_bounds=np.array([0.1, 0.55]),
    )


@pytest.fixture
def prediction_powered_result(sequence):
    return PredictionPoweredMeanMonitoringResult(
        metric_name="accuracy",
        monitor_name="PPI",
        higher_is_better=True,
        alarm_threshold=0.5,
        confidence_level=0.95,
        batch_mean_estimates=np.array([0.4, 0.8]),
        confidence_sequence=sequence,
        batch_n_true=np.array([10, 12]),
        batch_n_proxy=np.array([100, 120]),
    )


def test_prediction_powered_batch_n_properties(prediction_powered_result):
    np.testing.assert_array_equal(prediction_powered_result.batch_n_true, np.array([10, 12]))
    np.testing.assert_array_equal(prediction_powered_result.batch_n_proxy, np.array([100, 120]))


def test_prediction_powered_post_init_delegates_to_validation(sequence):
    batch_mean_estimates = np.array([0.4, 0.8])
    batch_n_true = np.array([10, 12])
    batch_n_proxy = np.array([100, 120])
    with patch.object(prediction_powered_module, "_validate_equal_lengths") as mock_validate_equal_lengths:
        PredictionPoweredMeanMonitoringResult(
            metric_name="accuracy",
            monitor_name="PPI",
            higher_is_better=True,
            alarm_threshold=0.5,
            confidence_level=0.95,
            batch_mean_estimates=batch_mean_estimates,
            confidence_sequence=sequence,
            batch_n_true=batch_n_true,
            batch_n_proxy=batch_n_proxy,
        )
        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], batch_mean_estimates)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], batch_n_true)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][2], batch_n_proxy)
        assert mock_validate_equal_lengths.call_args[1] == {
            "names": ["batch_mean_estimates", "batch_n_true", "batch_n_proxy"]
        }


def test_prediction_powered_str(prediction_powered_result):
    expected = (
        "Metric: accuracy\n"
        "Monitor: PPI\n"
        "Number of Batches: 2\n"
        "Drift Detected: True\n"
        "First Alarm Index: 0\n"
        "Alarm Threshold: 0.500\n"
        "Running Mean: 0.600\n"
        "Confidence Bound: 0.550\n"
        "Confidence Level: 0.95\n"
        "batch_n_true: [10, 12]\n"
        "batch_n_proxy: [100, 120]"
    )
    assert str(prediction_powered_result) == expected
