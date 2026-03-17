from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.inference_result import (
    ClassicalMeanInferenceResult,
    InferenceResult,
    MeanInferenceResult,
    SemiSupervisedMeanInferenceResult,
)

CI = CLTConfidenceInterval(mean=0.7, std=0.05, confidence_level=0.95)
RESULT = SemiSupervisedMeanInferenceResult(
    confidence_interval=CI,
    metric_name="accuracy",
    estimator_name="Test",
    n_true=10,
    n_proxy=90,
    effective_sample_size=200,
)


# --- SemiSupervisedMeanInferenceResult ---


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


def test_semi_supervised_str_contains_expected_fields():
    s = str(RESULT)
    assert "accuracy" in s
    assert "Test" in s
    assert "0.700" in s
    assert "10" in s
    assert "90" in s
    assert "200.0" in s
    assert "95" in s


def test_semi_supervised_width():
    assert RESULT.width == CI.upper_bound - CI.lower_bound


def test_semi_supervised_repr_equals_str_equals_summary():
    assert repr(RESULT) == str(RESULT)
    assert str(RESULT) == RESULT.summary()


# --- ClassicalMeanInferenceResult ---

CLASSICAL_CI = CLTConfidenceInterval(mean=0.6, std=0.1, confidence_level=0.9)
CLASSICAL_RESULT = ClassicalMeanInferenceResult(
    confidence_interval=CLASSICAL_CI,
    metric_name="precision",
    estimator_name="Classical",
    n=500,
)


def test_classical_attributes():
    assert CLASSICAL_RESULT.metric_name == "precision"
    assert CLASSICAL_RESULT.estimator_name == "Classical"
    assert CLASSICAL_RESULT.n == 500
    assert CLASSICAL_RESULT.mean == 0.6
    assert CLASSICAL_RESULT.std == 0.1


def test_classical_str_contains_expected_fields():
    s = str(CLASSICAL_RESULT)
    assert "precision" in s
    assert "Classical" in s
    assert "0.600" in s
    assert "500" in s
    assert "90" in s


def test_classical_repr_equals_str_equals_summary():
    assert repr(CLASSICAL_RESULT) == str(CLASSICAL_RESULT)
    assert str(CLASSICAL_RESULT) == CLASSICAL_RESULT.summary()


def test_classical_width():
    assert CLASSICAL_RESULT.width == CLASSICAL_CI.upper_bound - CLASSICAL_CI.lower_bound


# --- InferenceResult backward-compatibility alias ---


def test_inference_result_alias():
    assert InferenceResult is SemiSupervisedMeanInferenceResult


def test_inference_result_alias_instantiation():
    result = InferenceResult(
        confidence_interval=CI,
        metric_name="accuracy",
        estimator_name="Alias",
        n_true=5,
        n_proxy=45,
        effective_sample_size=100,
    )
    assert isinstance(result, SemiSupervisedMeanInferenceResult)
    assert result.estimator_name == "Alias"


# --- common lines shared formatting ---


def test_common_lines_confidence_level_formatting():
    # 95% CI should show "95" in the string
    s = str(RESULT)
    assert "95%" in s or "95" in s

    # 90% CI should show "90" in the string
    s_classical = str(CLASSICAL_RESULT)
    assert "90%" in s_classical or "90" in s_classical


def test_semi_supervised_str_format():
    s = str(RESULT)
    assert "Metric" in s
    assert "Point Estimate" in s
    assert "Confidence Interval" in s
    assert "Estimator" in s
    assert "n_true" in s
    assert "n_proxy" in s
    assert "Effective Sample Size" in s


def test_protocol_str_contains_common_lines():
    # Directly invoke MeanInferenceResult.__str__ to cover the Protocol's base implementation
    s = MeanInferenceResult.__str__(RESULT)
    assert "accuracy" in s
    assert "0.700" in s
    assert "Test" in s


def test_classical_str_format():
    s = str(CLASSICAL_RESULT)
    assert "Metric" in s
    assert "Point Estimate" in s
    assert "Confidence Interval" in s
    assert "Estimator" in s
    assert "n:" in s
