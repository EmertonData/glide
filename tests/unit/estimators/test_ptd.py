from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from glide.confidence_intervals import BootstrapConfidenceInterval
from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
from glide.estimators.ptd import PTDMeanEstimator

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def estimator() -> PTDMeanEstimator:
    return PTDMeanEstimator()


@pytest.fixture
def y_arrays() -> Tuple[NDArray, NDArray]:
    y_true = np.array([5.0, 6.0, 7.0, np.nan, np.nan, np.nan])
    y_proxy = np.array([4.5, 5.5, 6.5, 6.0, 7.0, 8.0])
    return y_true, y_proxy


@pytest.fixture
def y_data() -> Tuple[NDArray, NDArray, NDArray]:
    y_true = np.array([5.0, 6.0, 7.0])
    y_proxy_labeled = np.array([4.5, 5.5, 6.5])
    y_proxy_unlabeled = np.array([6.0, 7.0, 8.0])
    return y_true, y_proxy_labeled, y_proxy_unlabeled


# ── _preprocess ───────────────────────────────────────────────────────────────


def test_preprocess(estimator, y_arrays):
    y_true_all, y_proxy_all = y_arrays
    y_true, y_proxy_labeled, y_proxy_unlabeled = estimator._preprocess(y_true_all, y_proxy_all)
    assert len(y_true) == 3
    assert len(y_proxy_labeled) == 3
    assert len(y_proxy_unlabeled) == 3
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


# ── _compute_unlabeled_proxy_mean ─────────────────────────────────────────────


def test_compute_unlabeled_proxy_mean_known_value(estimator, y_data):
    _, _, y_proxy_unlabeled = y_data
    result = estimator._compute_unlabeled_proxy_mean(y_proxy_unlabeled)
    expected = 7.0
    assert result == pytest.approx(expected, abs=0.01)


# ── _compute_unlabeled_proxy_var ─────────────────────────────────────────


def test_compute_unlabeled_proxy_var_known_value(estimator, y_data):
    _, _, y_proxy_unlabeled = y_data
    result = estimator._compute_unlabeled_proxy_var(y_proxy_unlabeled)
    expected = 1.0 / 3.0
    assert result == pytest.approx(expected, abs=0.01)


# ── _compute_tuning_parameter ────────────────────────────────────────────────────


def test_compute_tuning_parameter_returns_one_when_power_tuning_false(estimator, y_data):
    y_true, y_proxy_labeled, y_proxy_unlabeled = y_data
    rng = np.random.default_rng(0)
    bootstraps = estimator._compute_bootstrap_labeled_estimates(y_true, y_proxy_labeled, 5, rng)
    var_proxy_unlabeled = estimator._compute_unlabeled_proxy_var(y_proxy_unlabeled)
    result = estimator._compute_tuning_parameter(bootstraps, var_proxy_unlabeled, power_tuning=False)
    assert result == 1.0


def test_compute_tuning_parameter_known_value(estimator, y_data):
    y_true, y_proxy_labeled, y_proxy_unlabeled = y_data
    rng = np.random.default_rng(42)
    bootstraps = estimator._compute_bootstrap_labeled_estimates(y_true, y_proxy_labeled, 5, rng)
    var_proxy_unlabeled = estimator._compute_unlabeled_proxy_var(y_proxy_unlabeled)
    result = estimator._compute_tuning_parameter(bootstraps, var_proxy_unlabeled, power_tuning=True)
    expected = 0.433
    assert result == pytest.approx(expected, abs=0.01)


# ── estimate ──────────────────────────────────────────────────────────────────


def test_estimate_returns_valid_inference_result(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, n_bootstrap=5, random_seed=0)
    assert isinstance(result, SemiSupervisedMeanInferenceResult)
    assert isinstance(result.confidence_interval, BootstrapConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "PTDMeanEstimator"


def test_estimate_metadata(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, n_bootstrap=5, metric_name="accuracy", random_seed=0)
    assert result.metric_name == "accuracy"
    assert result.estimator_name == "PTDMeanEstimator"
    assert result.n_true == 3
    assert result.n_proxy == 6
    assert result.effective_sample_size == 4


def test_estimate_custom_confidence_level(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(
        y_true, y_proxy, metric_name="perf", confidence_level=0.90, n_bootstrap=5, random_seed=0
    )

    expected_mean = 6.572
    expected_std = 0.453
    expected_lower = 6.176
    expected_upper = 7.152

    assert result.confidence_interval.confidence_level == 0.90
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


def test_estimate_determinism(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result_a = estimator.estimate(y_true, y_proxy, n_bootstrap=5, random_seed=7)
    result_b = estimator.estimate(y_true, y_proxy, n_bootstrap=5, random_seed=7)
    assert result_a.confidence_interval.lower_bound == result_b.confidence_interval.lower_bound
    assert result_a.confidence_interval.upper_bound == result_b.confidence_interval.upper_bound


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, n_bootstrap=5, metric_name="performance", random_seed=0)
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 6.572\n"
        "Confidence Interval (95%): [6.171, 7.192]\n"
        "Estimator : PTDMeanEstimator\n"
        "n_true: 3\n"
        "n_proxy: 6\n"
        "Effective Sample Size: 4"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_arrays):
    y_true, y_proxy = y_arrays
    result = estimator.estimate(y_true, y_proxy, n_bootstrap=5, metric_name="performance", random_seed=0)
    assert repr(result) == str(result)
