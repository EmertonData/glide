from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
from glide.estimators.asi import ASIMeanEstimator

# --- helpers ---


@pytest.fixture
def arrays(n_labeled: int = 2, n_unlabeled: int = 2, seed: int = 0) -> Tuple[NDArray, NDArray, NDArray]:
    rng = np.random.default_rng(seed)
    pi = n_labeled / (n_labeled + n_unlabeled)
    y_true_vals = rng.normal(4.0, 1.0, size=n_labeled)
    y_proxy_labeled = y_true_vals + rng.normal(0, 0.2, size=n_labeled)
    y_proxy_unlabeled = rng.normal(4.0, 1.0, size=n_unlabeled)
    y_true = np.hstack([y_true_vals, np.full(n_unlabeled, np.nan)])
    y_proxy = np.hstack([y_proxy_labeled, y_proxy_unlabeled])
    sampling_probabilities = np.full(n_labeled + n_unlabeled, pi)
    return y_true, y_proxy, sampling_probabilities


@pytest.fixture
def estimator() -> ASIMeanEstimator:
    return ASIMeanEstimator()


@pytest.fixture
def y_data() -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    y_true = np.array([3.0, 5.0, 0.0, 0.0])
    y_proxy = np.array([2.0, 4.0, 5.0, 7.0])
    xi = np.array([1.0, 1.0, 0.0, 0.0])
    pi = np.array([0.5, 0.5, 0.5, 0.5])
    y_data = (y_true, y_proxy, xi, pi)
    return y_data


@pytest.fixture
def rectified_labels() -> NDArray:
    return np.array([2.0, 4.0, 3.0, 5.0])


# --- preprocessing ---


def test_preprocess(estimator, arrays):
    y_true_input, y_proxy_input, sampling_probabilities = arrays
    y_true, y_proxy, xi, pi = estimator._preprocess(y_true_input, y_proxy_input, sampling_probabilities)
    assert len(y_true) == 4
    assert len(y_proxy) == 4
    assert len(xi) == 4
    assert int(xi.sum()) == 2
    assert len(pi) == 4
    assert np.all((pi > 0)) and np.all((pi <= 1))
    assert np.isin(xi, [0.0, 1.0]).all()
    assert not np.any(np.isnan(y_true))


def test_preprocess_raises_on_length_mismatch(estimator):
    y_true = np.array([1.0, np.nan])
    y_proxy = np.array([1.0, 1.0, 1.0])
    sampling_probabilities = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="same length"):
        estimator._preprocess(y_true, y_proxy, sampling_probabilities)


@pytest.mark.parametrize("bad_pi", [0.0, -0.5, 2.0])
def test_preprocess_raises_on_non_positive_pi(estimator, bad_pi):
    y_true = np.array([1.0, np.nan])
    y_proxy = np.array([1.0, 2.0])
    sampling_probabilities = np.array([0.5, bad_pi])
    with pytest.raises(ValueError, match="Sampling probabilities should be in \\(0, 1]"):
        estimator._preprocess(y_true, y_proxy, sampling_probabilities)


def test_preprocess_raises_on_constant_proxy(estimator):
    y_true = np.array([1.0, np.nan])
    y_proxy = np.array([1.0, 1.0])
    sampling_probabilities = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="Input proxy values have zero variance"):
        estimator._preprocess(y_true, y_proxy, sampling_probabilities)


def test_preprocess_raises_on_nan_proxy(estimator):
    y_true = np.array([1.0, np.nan])
    y_proxy = np.array([1.0, float("nan")])
    sampling_probabilities = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="Input proxy values contain NaN"):
        estimator._preprocess(y_true, y_proxy, sampling_probabilities)


# --- _compute_tuning_parameter ---


def test_compute_tuning_parameter_returns_one_when_power_tuning_false(estimator, y_data):
    # Hand-crafted arrays for deterministic unit tests.
    lam = estimator._compute_tuning_parameter(y_data, power_tuning=False)
    assert lam == 1.0


def test_compute_tuning_parameter_known_values(estimator, y_data):
    lam = estimator._compute_tuning_parameter(y_data, power_tuning=True)
    expected = 0.89
    assert lam == pytest.approx(expected, abs=0.01)


# --- _compute_mean_estimate ---


def test_compute_mean_estimate_known_values(estimator, rectified_labels):
    mean = estimator._compute_mean_estimate(rectified_labels)
    expected = 3.5
    assert mean == pytest.approx(expected)


# --- _compute_std_estimate ---


def test_compute_std_estimate_known_values(estimator, rectified_labels):
    std = estimator._compute_std_estimate(rectified_labels)
    expected = 0.65
    assert std == pytest.approx(expected, abs=0.01)


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, arrays):
    y_true, y_proxy, sampling_probabilities = arrays
    result = estimator.estimate(y_true, y_proxy, sampling_probabilities)
    assert isinstance(result, SemiSupervisedMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "ASIMeanEstimator"


def test_estimate_metadata(estimator, arrays):
    y_true, y_proxy, sampling_probabilities = arrays
    result = estimator.estimate(y_true, y_proxy, sampling_probabilities, metric_name="TestMetric")
    assert result.metric_name == "TestMetric"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 2
    assert result.n_proxy == 4
    assert result.effective_sample_size == 0


def test_estimate_custom_confidence_level(estimator, arrays):
    y_true, y_proxy, sampling_probabilities = arrays
    result = estimator.estimate(y_true, y_proxy, sampling_probabilities, confidence_level=0.95)

    expected_mean = 3.92
    expected_std = 0.19
    expected_lower = 3.55
    expected_upper = 4.28

    assert result.confidence_interval.confidence_level == 0.95
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, arrays):
    y_true, y_proxy, sampling_probabilities = arrays
    result = estimator.estimate(y_true, y_proxy, sampling_probabilities, metric_name="accuracy")
    output = str(result)
    expected = (
        "Metric: accuracy\n"
        "Point Estimate: 3.918\n"
        "Confidence Interval (95%): [3.555, 4.281]\n"
        "Estimator : ASIMeanEstimator\n"
        "n_true: 2\n"
        "n_proxy: 4\n"
        "Effective Sample Size: 0"
    )
    assert output == expected


def test_repr_equals_str(estimator, arrays):
    y_true, y_proxy, sampling_probabilities = arrays
    result = estimator.estimate(y_true, y_proxy, sampling_probabilities, metric_name="perf")
    assert repr(result) == str(result)
