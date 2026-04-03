import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.mean_inference_result import ClassicalMeanInferenceResult
from glide.estimators.stratified_classical import StratifiedClassicalMeanEstimator

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset() -> Dataset:
    """Two strata (A and B) with two records each.

    Stratum A: y=[1.0, 3.0]  → mean=2.0, std_mean=1.0
    Stratum B: y=[5.0, 7.0]  → mean=6.0, std_mean=1.0
    Weighted: mean=4.0, std=sqrt(0.5)≈0.707, n=4.
    """
    return Dataset(
        [
            {"y": 1.0, "group": "A"},
            {"y": 3.0, "group": "A"},
            {"y": 5.0, "group": "B"},
            {"y": 7.0, "group": "B"},
        ]
    )


@pytest.fixture
def estimator() -> StratifiedClassicalMeanEstimator:
    return StratifiedClassicalMeanEstimator()


# --- _get_strata ---


def test_get_strata_splits_correctly(estimator, dataset):
    strata = estimator._get_strata(dataset, "group")
    assert set(strata.keys()) == {"A", "B"}
    assert len(strata["A"]) == 2
    assert len(strata["B"]) == 2
    assert all(r["group"] == "A" for r in strata["A"])
    assert all(r["group"] == "B" for r in strata["B"])


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", groups_field="group")
    assert isinstance(result, ClassicalMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "StratifiedClassicalMeanEstimator"


def test_estimate_metadata(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", groups_field="group", metric_name="performance")
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n == 4


def test_estimate_known_values(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", groups_field="group", confidence_level=0.95)

    expected_mean = 4.0
    expected_std = np.sqrt(0.5)
    expected_lower = 2.614
    expected_upper = 5.386

    assert result.confidence_interval.confidence_level == 0.95
    assert result.confidence_interval.mean == pytest.approx(expected_mean)
    assert result.std == pytest.approx(expected_std)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower, abs=0.001)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper, abs=0.001)


# --- __str__ / __repr__ ---


def test_str_format(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", groups_field="group", metric_name="performance")
    output = str(result)
    expected = (
        "Metric: performance\n"
        "Point Estimate: 4.000\n"
        "Confidence Interval (95%): [2.614, 5.386]\n"
        "Estimator : StratifiedClassicalMeanEstimator\n"
        "n: 4"
    )
    assert output == expected


def test_repr_equals_str(estimator, dataset):
    result = estimator.estimate(dataset, y_field="y", groups_field="group", metric_name="perf")
    assert repr(result) == str(result)
