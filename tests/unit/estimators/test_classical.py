import numpy as np
import pytest
from numpy.typing import NDArray

from glide.core.dataset import Dataset
from glide.core.mean_inference_result import ClassicalMeanInferenceResult
from glide.estimators.classical import ClassicalMeanEstimator


@pytest.fixture
def dataset(n: int = 4, seed: int = 42) -> Dataset:
    rng = np.random.default_rng(seed)
    y = rng.normal(loc=5.0, scale=1.0, size=n)
    return Dataset([{"y": float(v)} for v in y])


@pytest.fixture
def estimator() -> ClassicalMeanEstimator:
    return ClassicalMeanEstimator()


@pytest.fixture
def y() -> NDArray:
    return np.array([2.0, 4.0, 6.0, 8.0])


# --- preprocessing ---


def test_preprocess(estimator, dataset):
    y = estimator._preprocess(dataset, "y")
    assert len(y) == 4


# --- _compute_mean_estimate ---


def test_compute_mean_estimate_known_values(estimator, y):
    expected = 5.0
    result = estimator._compute_mean_estimate(y)
    assert result == pytest.approx(expected)


# --- _compute_std_estimate ---


def test_compute_std_estimate_known_values(estimator, y):
    expected = 1.29
    result = estimator._compute_std_estimate(y)
    assert result == pytest.approx(expected, abs=0.01)


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y")
    assert isinstance(result, ClassicalMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "ClassicalMeanEstimator"


def test_estimate_metadata(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n == 4


def test_estimate_custom_confidence_level(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", confidence_level=0.90)
    assert result.confidence_interval.confidence_level == 0.90

    expected_mean = 5.24
    expected_std = 0.45
    expected_lower = 4.50
    expected_upper = 5.97

    assert result.confidence_interval.confidence_level == 0.90
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 5.239\n"
        "Confidence Interval (95%): [4.36, 6.11]\n"
        "Estimator : ClassicalMeanEstimator\n"
        "n: 4"
    )
    assert output == expected


def test_repr_equals_str(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", metric_name="perf")
    assert repr(result) == str(result)
