import pytest

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.inference_result import (
    ClassicalMeanInferenceResult,
    MeanInferenceResult,
)

# --- MeanInferenceResult (common attributes and properties) ---

_CI = CLTConfidenceInterval(mean=0.7, std=0.05, confidence_level=0.95)
_CLASSICAL = ClassicalMeanInferenceResult(
    confidence_interval=_CI,
    metric_name="accuracy",
    estimator_name="Test",
    n=100,
)


def test_base_mean():
    assert _CLASSICAL.mean == 0.7


def test_base_std():
    assert _CLASSICAL.std == 0.05


def test_base_width():
    assert _CLASSICAL.width == pytest.approx(0.1959963984540054)


def test_base_repr_equals_str_equals_summary():
    assert repr(_CLASSICAL) == str(_CLASSICAL)
    assert str(_CLASSICAL) == _CLASSICAL.summary()


# --- MeanInferenceResult.__str__ ---

_CLASSICAL_1 = ClassicalMeanInferenceResult(
    confidence_interval=CLTConfidenceInterval(mean=0.6, std=0.1, confidence_level=0.9),
    metric_name="metric1",
    estimator_name="Classical",
    n=500,
)


def test_base_classical_1():
    expected = "Metric: metric1\nPoint Estimate: 0.600\nConfidence Interval (90%): [0.44, 0.76]\nEstimator : Classical"
    assert MeanInferenceResult.__str__(_CLASSICAL_1) == expected
