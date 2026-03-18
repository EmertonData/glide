import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.mean_inference_result import ClassicalMeanInferenceResult
from glide.estimators.classical import ClassicalMeanEstimator


def make_dataset(n: int = 50, seed: int = 42) -> Dataset:
    rng = np.random.default_rng(seed)
    y = rng.normal(loc=5.0, scale=1.0, size=n)
    return Dataset([{"y": float(v)} for v in y])


@pytest.fixture
def dataset() -> Dataset:
    return make_dataset(n=50)


@pytest.fixture
def estimator() -> ClassicalMeanEstimator:
    return ClassicalMeanEstimator()


# --- preprocessing ---


def test_preprocess_returns_all_records(estimator, dataset):
    y = estimator._preprocess(dataset, "y")
    assert len(y) == 50


# --- _compute_mean_estimate ---


def test_classical_mean_known_values(estimator):
    y = np.array([2.0, 4.0, 6.0, 8.0])
    expected = 5.0
    assert estimator._compute_mean_estimate(y) == pytest.approx(expected)


# --- _compute_std_estimate ---


def test_classical_std_known_values(estimator):
    y = np.array([2.0, 4.0, 6.0, 8.0])
    expected = 1.2909944487358056
    assert estimator._compute_std_estimate(y) == pytest.approx(expected)


# --- estimate ---


def test_estimate_returns_classical_inference_result(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y")
    assert isinstance(result, ClassicalMeanInferenceResult)


def test_estimate_metadata(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n == 50


def test_estimate_custom_confidence_level(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", confidence_level=0.90)
    assert result.confidence_interval.confidence_level == 0.90


# --- __str__ / __repr__ ---


def test_str_format(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 5.091\n"
        "Confidence Interval (95%): [4.88, 5.30]\n"
        "Estimator : ClassicalMeanEstimator\n"
        "n: 50"
    )
    assert output == expected


def test_repr_equals_str(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", metric_name="perf")
    assert repr(result) == str(result)
