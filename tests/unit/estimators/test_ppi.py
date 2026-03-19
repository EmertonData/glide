import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
from glide.estimators.ppi import PPIMeanEstimator

# ── helpers ────────────────────────────────────────────────────────────────────


def make_dataset(n_true: int = 2, n_proxy: int = 2, seed: int = 42) -> Dataset:
    rng = np.random.default_rng(seed)
    y_true = rng.normal(loc=5.0, scale=1.0, size=n_true)
    y_proxy_labeled = y_true + rng.normal(0, 0.5, size=n_true)
    y_proxy_unlabeled = rng.normal(loc=5.0, scale=1.0, size=n_proxy) + rng.normal(0, 0.5, size=n_proxy)
    labeled = [{"y_true": float(y), "y_proxy": float(yh)} for y, yh in zip(y_true, y_proxy_labeled)]
    proxy_only = [{"y_proxy": float(yh)} for yh in y_proxy_unlabeled]
    return Dataset(labeled + proxy_only)


@pytest.fixture
def dataset() -> Dataset:
    return make_dataset(n_true=2, n_proxy=2)


@pytest.fixture
def estimator() -> PPIMeanEstimator:
    return PPIMeanEstimator()


# --- preprocessing ---


def test_preprocess_returns_tuple(estimator, dataset):
    y_data = estimator._preprocess(dataset, "y_true", "y_proxy")
    y_true, y_proxy_labeled, y_proxy_unlabeled = y_data
    assert len(y_true) == 2
    assert len(y_proxy_labeled) == 2
    assert len(y_proxy_unlabeled) == 2
    assert not np.any(np.isnan(y_true))


# --- _compute_lambda ---


def test_compute_lambda_returns_one_when_power_tuning_false(estimator, dataset):
    y_data = estimator._preprocess(dataset, "y_true", "y_proxy")
    result = estimator._compute_lambda(y_data, power_tuning=False)
    assert result == 1.0


def test_compute_lambda_known_values(estimator):
    y_true = np.array([0.0, 1.0])
    y_proxy_labeled = np.array([0.0, 1.0])
    y_proxy_unlabeled = np.array([0.0, 1.0])
    y_data = (y_true, y_proxy_labeled, y_proxy_unlabeled)
    expected = 0.75
    result = estimator._compute_lambda(y_data, power_tuning=True)
    assert result == pytest.approx(expected)


# --- _compute_mean_estimate ---


def test_ppi_mean_with_lambda_other(estimator):
    y_true = np.array([5.0, 6.0, 7.0])
    y_proxy_labeled = np.array([4.5, 5.5, 6.5])
    y_proxy_unlabeled = np.array([6.0, 7.0, 8.0])
    y_data = (y_true, y_proxy_labeled, y_proxy_unlabeled)
    expected = 6.75
    result = estimator._compute_mean_estimate(y_data, _lambda=0.5)
    assert result == pytest.approx(expected)


# --- _compute_std_estimate ---


def test_ppi_std_with_lambda_other(estimator):
    y_true = np.array([5.0, 6.0, 7.0])
    y_proxy_labeled = np.array([4.5, 5.5, 6.5])
    y_proxy_unlabeled = np.array([4.0, 5.0, 6.0, 7.0])
    y_data = (y_true, y_proxy_labeled, y_proxy_unlabeled)
    expected = 0.43
    result = estimator._compute_std_estimate(y_data, _lambda=0.5)
    assert result == pytest.approx(expected, abs=1e-2)


# --- estimate ---


def test_estimate_returns_semisupervised_mean_inference_result(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="perf")
    assert isinstance(result, SemiSupervisedMeanInferenceResult)


def test_estimate_metadata(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 2
    assert result.n_proxy == 4


def test_estimate_custom_confidence_level(estimator, dataset):
    result = estimator.estimate(
        dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="perf", confidence_level=0.90
    )
    assert result.confidence_interval.confidence_level == 0.90


# --- __str__ / __repr__ ---


def test_str_format(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="performance")
    output = str(result)
    assert "Metric: performance" in output
    assert "Point Estimate:" in output
    assert "Confidence Interval (95%):" in output
    assert f"Estimator : {estimator.__class__.__name__}" in output
    assert "n_true: 2" in output
    assert "n_proxy: 4" in output
    assert "Effective Sample Size:" in output


def test_repr_equals_str(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="perf")
    assert repr(result) == str(result)
