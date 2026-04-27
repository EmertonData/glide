from typing import Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.mean_inference_result import PredictionPoweredMeanInferenceResult
from glide.estimators.asi import ASIMeanEstimator

# --- helpers ---


@pytest.fixture
def y_arrays() -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    y_true = np.array([3.0, 5.0, np.nan, np.nan])
    y_proxy = np.array([2.0, 4.0, 5.0, 7.0])
    xi = np.array([1.0, 1.0, 0.0, 0.0])
    pi = np.array([0.5, 0.5, 0.5, 0.5])
    return y_true, y_proxy, xi, pi


@pytest.fixture
def estimator() -> ASIMeanEstimator:
    return ASIMeanEstimator()


# --- preprocessing ---


def test_preprocess_valid_output(estimator, y_arrays):
    y_true_input, y_proxy_input, _, pi = y_arrays
    y_true, y_proxy, xi, pi = estimator._preprocess(y_true_input, y_proxy_input, pi)
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
    pi = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="same length"):
        estimator._preprocess(y_true, y_proxy, pi)


@pytest.mark.parametrize("bad_pi", [0.0, -0.5, 2.0])
def test_preprocess_raises_on_non_positive_pi(estimator, bad_pi):
    y_true = np.array([1.0, np.nan])
    y_proxy = np.array([1.0, 2.0])
    pi = np.array([0.5, bad_pi])
    with pytest.raises(ValueError, match="Sampling probabilities should be in \\(0, 1]"):
        estimator._preprocess(y_true, y_proxy, pi)


def test_preprocess_raises_on_nan_proxy(estimator):
    y_true = np.array([1.0, np.nan])
    y_proxy = np.array([1.0, float("nan")])
    pi = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="Input proxy values contain NaN"):
        estimator._preprocess(y_true, y_proxy, pi)


def test_preprocess_raises_on_constant_proxy(estimator):
    y_true = np.array([1.0, np.nan])
    y_proxy = np.array([1.0, 1.0])
    pi = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="Input proxy values have zero variance"):
        estimator._preprocess(y_true, y_proxy, pi)


def test_preprocess_raises_on_unlabeled_samples_with_one_pi(estimator):
    y_true = np.array([1.0, np.nan, np.nan, np.nan])
    y_proxy = np.array([0.9, 1.9, 0.8, 1.8])
    pi = np.array([0.5, 1.0, 0.5, 0.5])
    with pytest.raises(ValueError, match="Samples with probability one of being labeled must be labeled"):
        estimator._preprocess(y_true, y_proxy, pi)


# --- _compute_tuning_parameter ---


def test_compute_tuning_parameter_returns_one_when_power_tuning_false(estimator, y_arrays):
    y_true_input, y_proxy, xi, pi = y_arrays
    y_true, y_proxy, xi, pi = estimator._preprocess(y_true_input, y_proxy, pi)
    lam = estimator._compute_tuning_parameter(y_true, y_proxy, xi, pi, power_tuning=False)
    assert lam == 1.0


def test_compute_tuning_parameter_known_values(estimator, y_arrays):
    y_true_input, y_proxy, xi, pi = y_arrays
    y_true, y_proxy, xi, pi = estimator._preprocess(y_true_input, y_proxy, pi)
    lam = estimator._compute_tuning_parameter(y_true, y_proxy, xi, pi, power_tuning=True)
    expected = 0.89
    assert lam == pytest.approx(expected, abs=0.01)


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, y_arrays):
    y_true, y_proxy, _, pi = y_arrays
    result = estimator.estimate(y_true, y_proxy, pi)
    assert isinstance(result, PredictionPoweredMeanInferenceResult)
    assert isinstance(result.confidence_interval, CLTConfidenceInterval)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "ASIMeanEstimator"


def test_estimate_metadata(estimator, y_arrays):
    y_true, y_proxy, _, pi = y_arrays
    result = estimator.estimate(y_true, y_proxy, pi, metric_name="TestMetric")
    assert result.metric_name == "TestMetric"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 2
    assert result.n_proxy == 4
    assert result.effective_sample_size == 5


def test_estimate_custom_confidence_level(estimator, y_arrays):
    y_true, y_proxy, _, pi = y_arrays
    result = estimator.estimate(y_true, y_proxy, pi, confidence_level=0.95)

    expected_mean = 5.34
    expected_std = 0.58
    expected_lower = 4.20
    expected_upper = 6.48

    assert result.confidence_interval.confidence_level == 0.95
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_arrays):
    y_true, y_proxy, _, pi = y_arrays
    result = estimator.estimate(y_true, y_proxy, pi, metric_name="accuracy")
    output = str(result)
    expected = (
        "Metric: accuracy\n"
        "Point Estimate: 5.341\n"
        "Confidence Interval (95%): [4.203, 6.479]\n"
        "Estimator : ASIMeanEstimator\n"
        "n_true: 2\n"
        "n_proxy: 4\n"
        "Effective Sample Size: 5"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_arrays):
    y_true, y_proxy, _, pi = y_arrays
    result = estimator.estimate(y_true, y_proxy, pi, metric_name="perf")
    assert repr(result) == str(result)
