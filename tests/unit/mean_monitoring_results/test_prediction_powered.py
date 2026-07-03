import numpy as np

from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.mean_monitoring_results import PredictionPoweredMeanMonitoringResult

# --- PredictionPoweredMeanMonitoringResult ---

_SEQUENCE = EmpiricalBernsteinConfidenceSequence(
    running_mean_estimates=np.array([0.4, 0.6]),
    confidence_bounds=np.array([0.1, 0.55]),
)
_PREDICTION_POWERED = PredictionPoweredMeanMonitoringResult(
    metric_name="accuracy",
    monitor_name="PPI",
    higher_is_better=True,
    alarm_threshold=0.5,
    confidence_level=0.95,
    batch_identifiers=np.array([0, 1]),
    batch_mean_estimates=np.array([0.4, 0.8]),
    confidence_sequence=_SEQUENCE,
    batch_n_true=np.array([10, 12]),
    batch_n_proxy=np.array([100, 120]),
)


def test_prediction_powered_batch_n_true():
    np.testing.assert_array_equal(_PREDICTION_POWERED.batch_n_true, np.array([10, 12]))


def test_prediction_powered_batch_n_proxy():
    np.testing.assert_array_equal(_PREDICTION_POWERED.batch_n_proxy, np.array([100, 120]))
