from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.inference_result import InferenceResult

CI = CLTConfidenceInterval(mean=0.7, std=0.05, confidence_level=0.95)
RESULT = InferenceResult(result=CI, metric_name="accuracy", estimator_name="Test", n_true=10, n_proxy=90)


def test_inference_result():
    assert RESULT.metric_name == "accuracy"
    assert RESULT.estimator_name == "Test"
    assert RESULT.n_true == 10
    assert RESULT.n_proxy == 90
    assert RESULT.result.mean == 0.7
    assert repr(RESULT) == str(RESULT)
    assert repr(RESULT) == RESULT.summary()
