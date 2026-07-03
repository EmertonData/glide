from unittest.mock import patch

import numpy as np
import pytest

import glide.mean_monitoring_results.classical as classical_module
from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult

# --- ClassicalMeanMonitoringResult ---


@pytest.fixture
def sequence():
    return EmpiricalBernsteinConfidenceSequence(
        running_mean_estimates=np.array([0.4, 0.6]),
        confidence_bounds=np.array([0.1, 0.55]),
    )


@pytest.fixture
def classical_result(sequence):
    return ClassicalMeanMonitoringResult(
        metric_name="accuracy",
        monitor_name="Classical",
        higher_is_better=True,
        alarm_threshold=0.5,
        confidence_level=0.95,
        batch_mean_estimates=np.array([0.4, 0.8]),
        confidence_sequence=sequence,
        batch_n=np.array([10, 12]),
    )


def test_classical_batch_n(classical_result):
    np.testing.assert_array_equal(classical_result.batch_n, np.array([10, 12]))


def test_classical_post_init_delegates_to_validation(sequence):
    batch_mean_estimates = np.array([0.4, 0.8])
    batch_n = np.array([10, 12])
    with patch.object(classical_module, "_validate_equal_lengths") as mock_validate_equal_lengths:
        ClassicalMeanMonitoringResult(
            metric_name="accuracy",
            monitor_name="Classical",
            higher_is_better=True,
            alarm_threshold=0.5,
            confidence_level=0.95,
            batch_mean_estimates=batch_mean_estimates,
            confidence_sequence=sequence,
            batch_n=batch_n,
        )
        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], batch_mean_estimates)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], batch_n)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["batch_mean_estimates", "batch_n"]}


def test_classical_str(classical_result):
    expected = (
        "Metric: accuracy\n"
        "Monitor: Classical\n"
        "Number of Batches: 2\n"
        "Drift Detected: True\n"
        "First Alarm Index: 0\n"
        "Alarm Threshold: 0.500\n"
        "Running Mean: 0.600\n"
        "Confidence Bound: 0.550\n"
        "Confidence Level: 0.95\n"
        "batch_n: 12"
    )
    assert str(classical_result) == expected
