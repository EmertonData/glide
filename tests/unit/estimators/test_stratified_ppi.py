import numpy as np
import pytest

from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
from glide.estimators.stratified_ppi import StratifiedPPIMeanEstimator

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def y_true() -> np.ndarray:
    """Ground-truth labels for two identical strata (A and B).

    Stratum A: labeled = [(5.0, 4.9), (6.0, 6.1)], unlabeled proxy = [5.2, 6.1]
    Stratum B: same as A.
    n_true = 4, n_proxy = 8.
    """
    return np.array([5.0, 6.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan])


@pytest.fixture
def y_proxy() -> np.ndarray:
    return np.array([4.9, 6.1, 5.2, 6.1, 4.9, 6.1, 5.2, 6.1])


@pytest.fixture
def groups() -> np.ndarray:
    return np.array(["A", "A", "A", "A", "B", "B", "B", "B"])


@pytest.fixture
def estimator() -> StratifiedPPIMeanEstimator:
    return StratifiedPPIMeanEstimator()


# --- tests ---


def test_estimate_raises_when_proxy_has_nan(estimator):
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([1.1, np.nan, 5.2, 6.1])  # NaN in proxy
    grps = np.array(["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="Input proxy values contain NaN"):
        estimator.estimate(y_true, y_proxy, grps)


def test_estimate_raises_when_stratum_has_too_few_labeled(estimator):
    # Each stratum has only 1 labeled and 1 unlabeled sample — too few for variance
    y_true = np.array([1.0, np.nan, 4.0, np.nan])
    y_proxy = np.array([1.1, 1.8, 3.9, 4.8])
    grps = np.array(["A", "A", "B", "B"])
    with pytest.raises(RuntimeError, match="Too few labeled or unlabeled samples in dataset stratum 'A'"):
        estimator.estimate(y_true, y_proxy, grps)


def test_estimate_raises_when_stratum_has_zero_proxy_variance(estimator):
    # Stratum B has constant proxy values
    y_true = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan])
    y_proxy = np.array([1.1, 2.1, 1.5, 1.5, 5.0, 5.0, 5.0, 5.0])  # Stratum B has zero variance
    grps = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
    with pytest.raises(ValueError, match="Input proxy values have zero variance in stratum 'B'"):
        estimator.estimate(y_true, y_proxy, grps)


def test_estimate_without_power_tuning(estimator, y_true, y_proxy, groups):
    # Test that power_tuning=False works (lambda_k = 1.0)
    result = estimator.estimate(y_true, y_proxy, groups, power_tuning=False)
    assert isinstance(result, SemiSupervisedMeanInferenceResult)
    assert np.isfinite(result.mean)
    assert np.isfinite(result.std)


def test_estimate_is_valid_inference_result(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups)
    assert isinstance(result, SemiSupervisedMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "StratifiedPPIMeanEstimator"


def test_estimate_metadata(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 4
    assert result.n_proxy == 8
    assert result.effective_sample_size == 5


def test_estimate_custom_confidence_level(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, confidence_level=0.85)

    expected_mean = 5.618
    expected_std = 0.250
    expected_lower = 5.257
    expected_upper = 5.978

    assert result.confidence_interval.confidence_level == 0.85
    assert result.confidence_interval.mean == pytest.approx(expected_mean, abs=0.01)
    assert result.std == pytest.approx(expected_std, abs=0.01)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.01)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.01)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 5.618\n"
        "Confidence Interval (95%): [5.127, 6.108]\n"
        "Estimator : StratifiedPPIMeanEstimator\n"
        "n_true: 4\n"
        "n_proxy: 8\n"
        "Effective Sample Size: 5"
    )
    assert output == expected


def test_repr_equals_str(estimator, y_true, y_proxy, groups):
    result = estimator.estimate(y_true, y_proxy, groups, metric_name="perf")
    assert repr(result) == str(result)
