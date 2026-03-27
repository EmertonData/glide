import numpy as np
import pytest
from numpy.typing import NDArray

from glide.core.dataset import Dataset
from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
from glide.estimators.ppi import PPIMeanEstimator

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset(n_true: int = 2, n_proxy: int = 2, seed: int = 42) -> Dataset:
    rng = np.random.default_rng(seed)
    y_true = rng.normal(loc=5.0, scale=1.0, size=n_true)
    y_proxy_labeled = y_true + rng.normal(0, 0.5, size=n_true)
    y_proxy_unlabeled = rng.normal(loc=5.0, scale=1.0, size=n_proxy) + rng.normal(0, 0.5, size=n_proxy)
    labeled = [{"y_true": float(y), "y_proxy": float(yh)} for y, yh in zip(y_true, y_proxy_labeled)]
    proxy_only = [{"y_proxy": float(yh)} for yh in y_proxy_unlabeled]
    return Dataset(labeled + proxy_only)


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


def test_preprocess(estimator, dataset):
    y_true, y_proxy_labeled, y_proxy_unlabeled = estimator._preprocess(dataset, "y_true", "y_proxy")
    assert len(y_true) == 2
    assert len(y_proxy_labeled) == 2
    assert len(y_proxy_unlabeled) == 2


def test_preprocess_no_nans_in_y_true(estimator, dataset):
    y_true, _, _ = estimator._preprocess(dataset, "y_true", "y_proxy")
    assert not np.any(np.isnan(y_true))


def test_preprocess_raises_when_only_one_sample(estimator):
    # Doctest has 2 labeled samples; keeping only 1 triggers the check
    labeled = [{"y_true": 5.0, "y_proxy": 4.9}]
    unlabeled = [{"y_proxy": 5.2}, {"y_proxy": 6.1}]
    dataset = Dataset(labeled + unlabeled)
    with pytest.raises(RuntimeError, match="Too few labeled or unlabeled samples in dataset"):
        estimator._preprocess(dataset, "y_true", "y_proxy")


def test_preprocess_raises_on_constant_proxy(estimator):
    labeled = [{"y_true": 1.0, "y_proxy": 1.0}]
    unlabeled = [{"y_proxy": 1.0}]
    dataset = Dataset(labeled + unlabeled)
    with pytest.raises(ValueError, match="Input proxy values have zero variance"):
        estimator._preprocess(dataset, "y_true", "y_proxy")


def test_preprocess_raises_on_nan_proxy(estimator):
    labeled = [{"y_true": 1.0, "y_proxy": 1.0}]
    unlabeled = [{"y_proxy": float("nan")}]
    dataset = Dataset(labeled + unlabeled)
    with pytest.raises(ValueError, match="Input proxy values contain NaN"):
        estimator._preprocess(dataset, "y_true", "y_proxy")


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


def test_estimate_is_valid_inference_result(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy")
    assert isinstance(result, SemiSupervisedMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "PPIMeanEstimator"


def test_estimate_metadata(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 2
    assert result.n_proxy == 4
    assert result.effective_sample_size == 4


def test_estimate_custom_confidence_level(estimator, dataset):
    result = estimator.estimate(
        dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="perf", confidence_level=0.90
    )

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


def test_str_format(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 4.068\n"
        "Confidence Interval (95%): [3.14, 5.00]\n"
        "Estimator : PPIMeanEstimator\n"
        "n_true: 2\n"
        "n_proxy: 4\n"
        "Effective Sample Size: 4.0"
    )
    assert output == expected


def test_repr_equals_str(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="perf")
    assert repr(result) == str(result)
