import math

import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.inference_result import ClassicalMeanInferenceResult
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


# --- _preprocess ---


def test_preprocess_returns_all_records(estimator, dataset):
    y = estimator._preprocess(dataset, "y")
    assert len(y) == 50


# --- _classical_mean ---


def test_classical_mean_known_values(estimator):
    y = np.array([2.0, 4.0, 6.0, 8.0])
    assert estimator._classical_mean(y) == pytest.approx(np.mean(y))


# --- _classical_std ---


def test_classical_std_known_values(estimator):
    y = np.array([2.0, 4.0, 6.0, 8.0])
    expected = np.std(y, ddof=1) / np.sqrt(len(y))
    assert estimator._classical_std(y) == pytest.approx(expected)


# --- estimate ---


def test_estimate_returns_classical_inference_result(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y")
    assert isinstance(result, ClassicalMeanInferenceResult)
    assert math.isfinite(result.confidence_interval.lower_bound)
    assert math.isfinite(result.confidence_interval.upper_bound)


def test_estimate_n_equals_record_count(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y")
    assert result.n == len(dataset)


def test_estimate_metadata(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == "ClassicalMeanEstimator"


def test_estimate_custom_confidence_level(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", confidence_level=0.90)
    assert result.confidence_interval.confidence_level == 0.90


def test_str_format(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", metric_name="performance")
    output = str(result)
    assert "Metric: performance" in output
    assert "Point Estimate:" in output
    assert "Confidence Interval (95%):" in output
    assert "Estimator : ClassicalMeanEstimator" in output
    assert "n: 50" in output
