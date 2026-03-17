from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.inference_result import SemiSupervisedMeanInferenceResult

CI = CLTConfidenceInterval(mean=0.7, std=0.05, confidence_level=0.95)
RESULT = SemiSupervisedMeanInferenceResult(
    confidence_interval=CI,
    metric_name="accuracy",
    estimator_name="Test",
    n_true=10,
    n_proxy=90,
    effective_sample_size=200,
)


def test_inference_result():
    assert RESULT.metric_name == "accuracy"
    assert RESULT.estimator_name == "Test"
    assert RESULT.n_true == 10
    assert RESULT.n_proxy == 90
    assert RESULT.mean == 0.7
    assert RESULT.std == 0.05
    assert RESULT.effective_sample_size == 200
    assert repr(RESULT) == str(RESULT)
    assert repr(RESULT) == RESULT.summary()
