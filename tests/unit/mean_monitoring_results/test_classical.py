import numpy as np

from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult

# --- ClassicalMeanMonitoringResult ---

_SEQUENCE = EmpiricalBernsteinConfidenceSequence(
    running_mean_estimates=np.array([0.4, 0.6]),
    confidence_bounds=np.array([0.1, 0.55]),
)
_CLASSICAL = ClassicalMeanMonitoringResult(
    metric_name="accuracy",
    monitor_name="Classical",
    higher_is_better=True,
    alarm_threshold=0.5,
    confidence_level=0.95,
    batch_identifiers=np.array([0, 1]),
    batch_mean_estimates=np.array([0.4, 0.8]),
    confidence_sequence=_SEQUENCE,
    batch_n=np.array([10, 12]),
)


def test_classical_batch_n():
    np.testing.assert_array_equal(_CLASSICAL.batch_n, np.array([10, 12]))
