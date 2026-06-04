import numpy as np
import pytest
from numpy.typing import NDArray

from glide.estimators import ClusterClassicalMeanEstimator
from glide.mean_inference_results import ClassicalMeanInferenceResult

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def y() -> NDArray:
    """Observations for two clusters (A and B) with two samples each.

    Cluster A: y=[5.0, 5.0]  → mean=5.0, sum=10.0
    Cluster B: y=[7.0, 7.0]  → mean=7.0, sum=14.0
    Weighted: mean=6.0, var=2*Var([10,14], ddof=1)/16=1.0, std=1.0, n=4.
    """
    return np.array([5.0, 5.0, 7.0, 7.0])


@pytest.fixture
def clusters() -> NDArray:
    return np.array(["A", "A", "B", "B"])


@pytest.fixture
def estimator() -> ClusterClassicalMeanEstimator:
    return ClusterClassicalMeanEstimator()


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, y, clusters):
    result = estimator.estimate(y, clusters)
    assert isinstance(result, ClassicalMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "ClusterClassicalMeanEstimator"


def test_estimate_metadata(estimator, y, clusters):
    result = estimator.estimate(y, clusters, metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n == 4


def test_estimate_analytical_values(estimator, y, clusters):
    result = estimator.estimate(y, clusters)

    expected_mean = 6.0
    expected_std = 1.0
    expected_lower = 4.040
    expected_upper = 7.960

    assert result.confidence_interval.mean == pytest.approx(expected_mean)
    assert result.std == pytest.approx(expected_std)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.001)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.001)


def test_estimate_custom_confidence_level(estimator, y, clusters):
    result = estimator.estimate(y, clusters, confidence_level=0.85)

    expected_lower = 4.560
    expected_upper = 7.440

    assert result.confidence_interval.confidence_level == 0.85
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.001)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.001)


def test_estimate_ignores_nans(estimator, y, clusters):
    y_with_nans = np.hstack([y, np.full(2, np.nan)])
    clusters_with_nans = np.hstack([clusters, np.array(["A", "B"])])

    result = estimator.estimate(y, clusters, metric_name="performance")
    result_with_nans = estimator.estimate(y_with_nans, clusters_with_nans, metric_name="performance")
    assert result == result_with_nans


def test_estimate_n_counts_non_nan_observations(estimator, y, clusters):
    y_with_nans = np.hstack([y, np.array([np.nan, np.nan])])
    clusters_with_nans = np.hstack([clusters, np.array(["A", "B"])])

    result = estimator.estimate(y_with_nans, clusters_with_nans)
    assert result.n == 4


def test_estimate_skips_all_nan_cluster(estimator, y, clusters):
    y_augmented = np.hstack([y, np.array([np.nan, np.nan])])
    clusters_augmented = np.hstack([clusters, np.array(["C", "C"])])

    result_base = estimator.estimate(y, clusters)
    result_augmented = estimator.estimate(y_augmented, clusters_augmented)
    assert result_base == result_augmented


def test_estimate_raises_for_fewer_than_two_valid_clusters(estimator):
    y = np.array([5.0, 7.0, np.nan, np.nan])
    clusters = np.array(["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="Need at least 2 clusters"):
        estimator.estimate(y, clusters)


def test_estimate_raises_when_y_and_clusters_have_different_lengths(estimator):
    y = np.array([1.0, 2.0])
    clusters = np.array(["A", "B", "C"])
    with pytest.raises(ValueError, match="same length"):
        estimator.estimate(y, clusters)


# --- __str__ / __repr__ ---


def test_str_format(estimator, y, clusters):
    result = estimator.estimate(y, clusters, metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 6.000\n"
        "Confidence Interval (95%): [4.040, 7.960]\n"
        "Estimator : ClusterClassicalMeanEstimator\n"
        "n: 4"
    )
    assert output == expected


def test_repr_equals_str(estimator, y, clusters):
    result = estimator.estimate(y, clusters, metric_name="perf")
    assert repr(result) == str(result)
