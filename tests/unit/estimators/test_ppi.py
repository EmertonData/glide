import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.inference_result import InferenceResult
from glide.estimators.ppi import PPIMeanEstimator


def make_dataset(n_true: int = 25, n_proxy: int = 75, seed: int = 42) -> Dataset:
    rng = np.random.default_rng(seed)
    y_true = rng.normal(loc=5.0, scale=1.0, size=n_true)
    y_proxy_labeled = y_true + rng.normal(0, 0.5, size=n_true)
    y_proxy_unlabeled = rng.normal(loc=5.0, scale=1.0, size=n_proxy) + rng.normal(0, 0.5, size=n_proxy)
    labeled = [{"y_true": float(y), "y_proxy": float(yh)} for y, yh in zip(y_true, y_proxy_labeled)]
    proxy_only = [{"y_proxy": float(yh)} for yh in y_proxy_unlabeled]
    return Dataset(labeled + proxy_only)


@pytest.fixture
def dataset() -> Dataset:
    return make_dataset(n_true=25, n_proxy=75)


@pytest.fixture
def estimator() -> PPIMeanEstimator:
    return PPIMeanEstimator()


# --- preprocessing ---


def test_preprocess_counts(estimator, dataset):
    y_true, y_proxy_labeled, y_proxy_unlabeled = estimator._preprocess(dataset, "y_true", "y_proxy")
    assert len(y_true) == 25
    assert len(y_proxy_labeled) == 25
    assert len(y_proxy_unlabeled) == 75


def test_preprocess_no_nans_in_y_true(estimator, dataset):
    y_true, _, _ = estimator._preprocess(dataset, "y_true", "y_proxy")
    assert not np.any(np.isnan(y_true))


# --- ppi_mean ---


def test_ppi_mean_matches_manual(estimator):
    y_true = np.array([5.0, 6.0, 7.0])
    y_proxy_labeled = np.array([4.5, 5.5, 6.5])
    y_proxy = np.array([4.0, 5.0, 6.0, 7.0])
    expected = 6.0
    result = estimator._ppi_mean((y_true, y_proxy_labeled, y_proxy))
    assert result == pytest.approx(expected)


# --- ppi_std ---


def test_ppi_std_matches_manual(estimator):
    y_true = np.array([5.0, 6.0, 7.0])
    y_proxy_labeled = np.array([4.5, 5.5, 6.5])
    y_proxy = np.array([4.0, 5.0, 6.0, 7.0])
    expected_std = 0.645497224
    result = estimator._ppi_std((y_true, y_proxy_labeled, y_proxy))
    assert result == pytest.approx(expected_std)


# --- ess ---


def test_ess_manual(estimator):
    y_true = np.array([5.0, 6.0, 7.0])
    y_proxy_labeled = np.array([4.5, 5.5, 6.5])
    y_proxy_unlabeled = np.array([4.0, 5.0, 6.0, 7.0])
    std = estimator._ppi_std(y_true, y_proxy_labeled, y_proxy_unlabeled)
    n = len(y_true)
    var_y_true = np.var(y_true, ddof=1)
    std_labeled = np.sqrt(var_y_true / n)
    ess = n * (std_labeled / std) ** 2
    assert ess == pytest.approx(2.4)


def test_ess_in_estimate_result(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy")
    assert result.ess is not None
    assert result.ess > 0


# --- estimate ---


def test_estimate_returns_inference_result(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="perf")
    assert isinstance(result, InferenceResult)


def test_estimate_metadata(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 25
    assert result.n_proxy == 100


def test_estimate_custom_confidence_level(estimator, dataset):
    result = estimator.estimate(
        dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="perf", confidence_level=0.90
    )
    assert result.result.confidence_level == 0.90


# --- __str__ / __repr__ ---


def test_str_format(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="performance")
    output = str(result)
    assert "Metric: performance" in output
    assert "Point Estimate:" in output
    assert "Confidence Interval (95%):" in output
    assert f"Estimator : {estimator.__class__.__name__}" in output
    assert "n_true: 25" in output
    assert "n_proxy: 100" in output
    assert "ESS:" in output


def test_repr_equals_str(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", metric_name="perf")
    assert repr(result) == str(result)
