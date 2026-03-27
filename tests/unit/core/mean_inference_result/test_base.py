import pytest

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.mean_inference_result import MeanInferenceResult

# --- MeanInferenceResult (common attributes and properties) ---

_CI = CLTConfidenceInterval(mean=0.7, std=0.05, confidence_level=0.95)
_MEAN_INFERENCE_RESULT = MeanInferenceResult(
    confidence_interval=_CI,
    metric_name="accuracy",
    estimator_name="Test",
)


def test_base_mean():
    assert _MEAN_INFERENCE_RESULT.mean == 0.7


def test_base_std():
    assert _MEAN_INFERENCE_RESULT.std == 0.05


def test_base_width():
    assert _MEAN_INFERENCE_RESULT.width == pytest.approx(0.1959963984540054)


def test_base_repr_equals_str_equals_summary():
    assert repr(_MEAN_INFERENCE_RESULT) == str(_MEAN_INFERENCE_RESULT)
    assert str(_MEAN_INFERENCE_RESULT) == _MEAN_INFERENCE_RESULT.summary()


# --- MeanInferenceResult.__str__ ---

_MEAN_INFERENCE_RESULT_1 = MeanInferenceResult(
    confidence_interval=CLTConfidenceInterval(mean=0.6, std=0.1, confidence_level=0.9),
    metric_name="metric1",
    estimator_name="Classical",
)


def test_base_classical_1():
    expected = (
        "Metric: metric1\nPoint Estimate: 0.600\nConfidence Interval (90%): [0.436, 0.764]\nEstimator : Classical"
    )
    assert MeanInferenceResult.__str__(_MEAN_INFERENCE_RESULT_1) == expected
