import numpy as np
import pytest
from numpy.typing import NDArray

from glide.core.mean_inference_result import ClassicalMeanInferenceResult
from glide.estimators.stratified_classical import StratifiedClassicalMeanEstimator

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def y() -> NDArray:
    """Observations for two strata (A and B) with two samples each.

    Stratum A: y=[1.0, 3.0]  → mean=2.0, std_mean=1.0
    Stratum B: y=[5.0, 7.0]  → mean=6.0, std_mean=1.0
    Weighted: mean=4.0, std=sqrt(0.5)≈0.707, n=4.
    """
    return np.array([1.0, 3.0, 5.0, 7.0])


@pytest.fixture
def groups() -> NDArray:
    return np.array(["A", "A", "B", "B"])


@pytest.fixture
def estimator() -> StratifiedClassicalMeanEstimator:
    return StratifiedClassicalMeanEstimator()


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, y, groups):
    result = estimator.estimate(y, groups)
    assert isinstance(result, ClassicalMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "StratifiedClassicalMeanEstimator"


def test_estimate_metadata(estimator, y, groups):
    result = estimator.estimate(y, groups, metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n == 4


def test_estimate_custom_confidence_level(estimator, y, groups):
    result = estimator.estimate(y, groups, confidence_level=0.85)

    expected_mean = 4.0
    expected_std = np.sqrt(0.5)
    expected_lower = 2.982
    expected_upper = 5.018

    assert result.confidence_interval.confidence_level == 0.85
    assert result.confidence_interval.mean == pytest.approx(expected_mean)
    assert result.std == pytest.approx(expected_std)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.001)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.001)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y, groups):
    result = estimator.estimate(y, groups, metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 4.000\n"
        "Confidence Interval (95%): [2.614, 5.386]\n"
        "Estimator : StratifiedClassicalMeanEstimator\n"
        "n: 4"
    )
    assert output == expected


def test_repr_equals_str(estimator, y, groups):
    result = estimator.estimate(y, groups, metric_name="perf")
    assert repr(result) == str(result)
