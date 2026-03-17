import pytest

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.inference_result import (
    ClassicalMeanInferenceResult,
    MeanInferenceResultBase,
    SemiSupervisedMeanInferenceResult,
)

# --- MeanInferenceResultBase (common attributes and properties) ---

_CI = CLTConfidenceInterval(mean=0.7, std=0.05, confidence_level=0.95)
_BASE = ClassicalMeanInferenceResult(
    confidence_interval=_CI,
    metric_name="accuracy",
    estimator_name="Test",
    n=100,
)


def test_base_mean():
    assert _BASE.mean == 0.7


def test_base_std():
    assert _BASE.std == 0.05


def test_base_width():
    assert _BASE.width == pytest.approx(0.1959963984540054)


def test_base_repr_equals_str_equals_summary():
    assert repr(_BASE) == str(_BASE)
    assert str(_BASE) == _BASE.summary()


# --- MeanInferenceResultBase.__str__ ---

_BASE_STR_1 = ClassicalMeanInferenceResult(
    confidence_interval=CLTConfidenceInterval(mean=0.6, std=0.1, confidence_level=0.9),
    metric_name="metric1",
    estimator_name="Classical",
    n=500,
)
_BASE_STR_2 = SemiSupervisedMeanInferenceResult(
    confidence_interval=CLTConfidenceInterval(mean=0.7, std=0.05, confidence_level=0.95),
    metric_name="accuracy",
    estimator_name="PPI",
    n_true=10,
    n_proxy=90,
    effective_sample_size=200,
)


def test_base_str_metric1_classical():
    expected = "Metric: metric1\nPoint Estimate: 0.600\nConfidence Interval (90%): [0.44, 0.76]\nEstimator : Classical"
    assert MeanInferenceResultBase.__str__(_BASE_STR_1) == expected


def test_base_str_accuracy_ppi():
    expected = "Metric: accuracy\nPoint Estimate: 0.700\nConfidence Interval (95%): [0.60, 0.80]\nEstimator : PPI"
    assert MeanInferenceResultBase.__str__(_BASE_STR_2) == expected


# --- ClassicalMeanInferenceResult ---

_CLASSICAL_1 = ClassicalMeanInferenceResult(
    confidence_interval=CLTConfidenceInterval(mean=0.6, std=0.1, confidence_level=0.9),
    metric_name="metric1",
    estimator_name="Classical",
    n=500,
)
_CLASSICAL_2 = ClassicalMeanInferenceResult(
    confidence_interval=CLTConfidenceInterval(mean=0.5, std=0.02, confidence_level=0.95),
    metric_name="metric2",
    estimator_name="Bootstrap",
    n=1000,
)


def test_classical_str_metric1():
    expected = (
        "Metric: metric1\nPoint Estimate: 0.600\nConfidence Interval (90%): [0.44, 0.76]\nEstimator : Classical\nn: 500"
    )
    assert str(_CLASSICAL_1) == expected


def test_classical_str_metric2():
    expected = (
        "Metric: metric2\nPoint Estimate: 0.500\nConfidence Interval (95%):"
        " [0.46, 0.54]\nEstimator : Bootstrap\nn: 1000"
    )
    assert str(_CLASSICAL_2) == expected


# --- SemiSupervisedMeanInferenceResult ---

_SEMI_SUPERVISED_1 = SemiSupervisedMeanInferenceResult(
    confidence_interval=CLTConfidenceInterval(mean=0.7, std=0.05, confidence_level=0.95),
    metric_name="accuracy",
    estimator_name="PPI",
    n_true=10,
    n_proxy=90,
    effective_sample_size=200,
)
_SEMI_SUPERVISED_2 = SemiSupervisedMeanInferenceResult(
    confidence_interval=CLTConfidenceInterval(mean=0.8, std=0.03, confidence_level=0.99),
    metric_name="f1",
    estimator_name="IPW",
    n_true=50,
    n_proxy=200,
    effective_sample_size=500,
)


def test_semi_supervised_str_accuracy():
    expected = (
        "Metric: accuracy\n"
        "Point Estimate: 0.700\n"
        "Confidence Interval (95%): [0.60, 0.80]\n"
        "Estimator : PPI\n"
        "n_true: 10\n"
        "n_proxy: 90\n"
        "Effective Sample Size: 200.0"
    )
    assert str(_SEMI_SUPERVISED_1) == expected


def test_semi_supervised_str_f1():
    expected = (
        "Metric: f1\n"
        "Point Estimate: 0.800\n"
        "Confidence Interval (99%): [0.72, 0.88]\n"
        "Estimator : IPW\n"
        "n_true: 50\n"
        "n_proxy: 200\n"
        "Effective Sample Size: 500.0"
    )
    assert str(_SEMI_SUPERVISED_2) == expected
