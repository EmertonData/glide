from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.inference_result import ClassicalMeanInferenceResult

# --- ClassicalMeanInferenceResult ---

_CLASSICAL = ClassicalMeanInferenceResult(
    confidence_interval=CLTConfidenceInterval(mean=0.6, std=0.1, confidence_level=0.9),
    metric_name="metric1",
    estimator_name="Classical",
    n=500,
)


def test_classical():
    expected = (
        "Metric: metric1\nPoint Estimate: 0.600\nConfidence Interval (90%): [0.44, 0.76]\nEstimator : Classical\nn: 500"
    )
    assert str(_CLASSICAL) == expected
