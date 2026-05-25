import numpy as np
import pytest

from glide.confidence_intervals import BootstrapConfidenceInterval
from glide.estimators import StratifiedPTDMeanEstimator
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def y_true() -> np.ndarray:
    return np.array([5.0, 6.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan])


@pytest.fixture
def y_proxy() -> np.ndarray:
    return np.array([4.9, 6.1, 5.2, 6.1, 4.9, 6.1, 5.2, 6.1])


@pytest.fixture
def groups() -> np.ndarray:
    return np.array(["A", "A", "A", "A", "B", "B", "B", "B"])


@pytest.fixture
def estimator() -> StratifiedPTDMeanEstimator:
    return StratifiedPTDMeanEstimator()


# --- estimate ---


def test_estimate_returns_valid_inference_result(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, n_bootstrap=5, random_seed=0)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, BootstrapConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "StratifiedPTDMeanEstimator"


def test_estimate_metadata(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, metric_name="accuracy", n_bootstrap=5, random_seed=0)
    assert result.metric_name == "accuracy"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 4
    assert result.n_proxy == 8
    assert result.effective_sample_size == 33


def test_estimate_custom_confidence_level(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, confidence_level=0.90, n_bootstrap=5, random_seed=0)

    expected_mean = 5.578
    expected_std = 0.122
    expected_lower = 5.414
    expected_upper = 5.663

    assert result.confidence_interval.confidence_level == 0.90
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


def test_estimate_reproducibility(estimator, y_true, y_proxy, groups):
    result_a = estimator.estimate(y_true, y_proxy, groups, n_bootstrap=5, random_seed=7)
    result_b = estimator.estimate(y_true, y_proxy, groups, n_bootstrap=5, random_seed=7)
    assert result_a.confidence_interval.lower_bound == result_b.confidence_interval.lower_bound
    assert result_a.confidence_interval.upper_bound == result_b.confidence_interval.upper_bound


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, metric_name="performance", n_bootstrap=5, random_seed=0)
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 5.578\n"
        "Confidence Interval (95%): [5.400, 5.664]\n"
        "Estimator : StratifiedPTDMeanEstimator\n"
        "n_true: 4\n"
        "n_proxy: 8\n"
        "Effective Sample Size: 33"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, metric_name="performance", n_bootstrap=5, random_seed=0)
    assert repr(result) == str(result)
