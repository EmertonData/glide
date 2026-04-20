from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.mean_inference_result import PredictionPoweredMeanInferenceResult

# --- PredictionPoweredMeanInferenceResult ---

_PREDICTION_POWERED = PredictionPoweredMeanInferenceResult(
    confidence_interval=CLTConfidenceInterval(mean=0.7, std=0.05, confidence_level=0.95),
    metric_name="accuracy",
    estimator_name="PPI",
    n_true=10,
    n_proxy=90,
    effective_sample_size=200,
)


def test_prediction_powered_str():
    expected = (
        "Metric: accuracy\n"
        "Point Estimate: 0.700\n"
        "Confidence Interval (95%): [0.602, 0.798]\n"
        "Estimator : PPI\n"
        "n_true: 10\n"
        "n_proxy: 90\n"
        "Effective Sample Size: 200"
    )
    assert str(_PREDICTION_POWERED) == expected
