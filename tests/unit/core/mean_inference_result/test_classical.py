from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.mean_inference_result import ClassicalMeanInferenceResult

# --- ClassicalMeanInferenceResult ---

_CLASSICAL = ClassicalMeanInferenceResult(
    confidence_interval=CLTConfidenceInterval(mean=0.6, std=0.1, confidence_level=0.9),
    metric_name="metric1",
    estimator_name="Classical",
    n=500,
)


def test_classical():
    expected = (
        "Metric: metric1\nPoint Estimate: 0.600\nConfidence Interval (90%): [0.436, 0.764]\nEstimator : Classical\n"
        "n: 500"
    )
    assert str(_CLASSICAL) == expected
