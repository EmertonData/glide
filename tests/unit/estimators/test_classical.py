import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.inference_result import InferenceResult
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


def test_estimate_returns_inference_result(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y")
    assert isinstance(result, InferenceResult)


def test_estimate_metadata(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == "ClassicalMeanEstimator"
    assert result.n_true == 50
    assert result.n_proxy == 0


def test_estimate_custom_confidence_level(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", confidence_level=0.90)
    assert result.result.confidence_level == 0.90


def test_estimate_mean_matches_manual(estimator):
    y = np.array([2.0, 4.0, 6.0, 8.0])
    dataset = Dataset([{"y": float(v)} for v in y])
    result = estimator.estimate(dataset, y_field="y")
    assert result.result.mean == pytest.approx(5.0)
    expected_std = 1.2909944487358056
    assert result.result.std == pytest.approx(expected_std)
    assert result.effective_sample_size == len(y)


def test_estimate_mean_nan_values_ignored(estimator):
    # Same values as test_estimate_mean_matches_manual but with NaN entries mixed in.
    # The estimator uses nanmean/nanstd, so the result must be identical.
    y = np.array([2.0, 4.0, 6.0, 8.0])
    y_with_nans = np.array([2.0, np.nan, 4.0, 6.0, np.nan, 8.0])
    dataset = Dataset([{"y": float(v)} for v in y_with_nans])
    result = estimator.estimate(dataset, y_field="y")
    assert result.result.mean == pytest.approx(np.nanmean(y))
    expected_std = np.std(y, ddof=1) / np.sqrt(len(y))
    assert result.result.std == pytest.approx(expected_std)


def test_str_format(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", metric_name="performance")
    output = str(result)
    assert "Metric: performance" in output
    assert "Point Estimate:" in output
    assert "Confidence Interval (95%):" in output
    assert "Estimator : ClassicalMeanEstimator" in output
    assert "n_true: 50" in output
    assert "n_proxy: 0" in output
    assert "Effective Sample Size:" in output
