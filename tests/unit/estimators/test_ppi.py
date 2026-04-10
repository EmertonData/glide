import numpy as np
import pytest
from numpy.typing import NDArray

from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
from glide.estimators.ppi import PPIMeanEstimator

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def y_arrays(n_true: int = 2, n_proxy: int = 2, seed: int = 42) -> tuple[NDArray, NDArray]:
    rng = np.random.default_rng(seed)
    y_true = rng.normal(loc=5.0, scale=1.0, size=n_true)
    y_proxy_labeled = y_true + rng.normal(0, 0.5, size=n_true)
    y_proxy_unlabeled = rng.normal(loc=5.0, scale=1.0, size=n_proxy) + rng.normal(0, 0.5, size=n_proxy)
    y_true_all = np.concatenate([y_true, np.full(n_proxy, np.nan)])
    y_proxy_all = np.concatenate([y_proxy_labeled, y_proxy_unlabeled])
    return y_true_all, y_proxy_all


@pytest.fixture
def estimator() -> PPIMeanEstimator:
    return PPIMeanEstimator()


@pytest.fixture
def y_data() -> tuple[NDArray, NDArray, NDArray]:
    y_true = np.array([5.0, 6.0, 7.0])
    y_proxy_labeled = np.array([4.5, 5.5, 6.5])
    y_proxy_unlabeled = np.array([6.0, 7.0, 8.0])
    return (y_true, y_proxy_labeled, y_proxy_unlabeled)


# --- preprocessing ---


def test_preprocess(estimator, y_arrays):
    y_true_all, y_proxy_all = y_arrays
    y_true, y_proxy_labeled, y_proxy_unlabeled = estimator._preprocess(y_true_all, y_proxy_all)
    assert len(y_true) == 2
    assert len(y_proxy_labeled) == 2
    assert len(y_proxy_unlabeled) == 2
    assert not np.any(np.isnan(y_true))


def test_preprocess_raises_when_only_one_sample(estimator):
    y_true = np.array([5.0, np.nan, np.nan])
    y_proxy = np.array([4.9, 5.2, 6.1])
    with pytest.raises(RuntimeError, match="Too few labeled or unlabeled samples in dataset"):
        estimator._preprocess(y_true, y_proxy)


def test_preprocess_raises_on_constant_proxy(estimator):
    y_true = np.array([1.0, np.nan])
    y_proxy = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="Input proxy values have zero variance"):
        estimator._preprocess(y_true, y_proxy)


def test_preprocess_raises_on_nan_proxy(estimator):
    y_true = np.array([1.0, np.nan])
    y_proxy = np.array([1.0, np.nan])
    with pytest.raises(ValueError, match="Input proxy values contain NaN"):
        estimator._preprocess(y_true, y_proxy)


def test_preprocess_raises_on_length_mismatch(estimator):
    y_true = np.array([1.0, 2.0, np.nan])
    y_proxy = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="y_true and y_proxy must have the same length"):
        estimator._preprocess(y_true, y_proxy)


# --- _compute_lambda ---


def test_compute_lambda_returns_one_when_power_tuning_false(estimator, y_data):
    result = estimator._compute_lambda(y_data, power_tuning=False)
    assert result == 1.0


def test_compute_lambda_known_values(estimator, y_data):
    expected = 0.34
    result = estimator._compute_lambda(y_data, power_tuning=True)
    assert result == pytest.approx(expected, abs=0.01)


# --- _compute_mean_estimate ---


def test_compute_mean_estimate_known_values(estimator, y_data):
    expected = 6.75
    result = estimator._compute_mean_estimate(y_data, _lambda=0.5)
    assert result == pytest.approx(expected)


# --- _compute_std_estimate ---


def test_compute_std_estimate_known_values(estimator, y_data):
    expected = 0.41
    result = estimator._compute_std_estimate(y_data, _lambda=0.5)
    assert result == pytest.approx(expected, abs=1e-2)


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy)
    assert isinstance(result, SemiSupervisedMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "PPIMeanEstimator"


def test_estimate_metadata(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 2
    assert result.n_proxy == 4
    assert result.effective_sample_size == 4


def test_estimate_custom_confidence_level(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="perf", confidence_level=0.90)

    expected_mean = 4.07
    expected_std = 0.47
    expected_lower = 3.29
    expected_upper = 4.85

    assert result.confidence_interval.confidence_level == 0.90
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 4.068\n"
        "Confidence Interval (95%): [3.140, 4.996]\n"
        "Estimator : PPIMeanEstimator\n"
        "n_true: 2\n"
        "n_proxy: 4\n"
        "Effective Sample Size: 4"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, metric_name="perf")
    assert repr(result) == str(result)
