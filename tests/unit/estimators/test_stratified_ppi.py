import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
from glide.estimators.stratified_ppi import StratifiedPPIMeanEstimator

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def dataset() -> Dataset:
    """Two identical strata (A and B), each matching the PPI doctest data.

    Stratum A: labeled = [(5.0, 4.9), (6.0, 6.1)], unlabeled = [5.2, 6.1]
    Stratum B: same records, different group value.
    n_true = 4, n_proxy = 8.
    """
    return Dataset(
        [
            {"y_true": 5.0, "y_proxy": 4.9, "group": "A"},
            {"y_true": 6.0, "y_proxy": 6.1, "group": "A"},
            {"y_proxy": 5.2, "group": "A"},
            {"y_proxy": 6.1, "group": "A"},
            {"y_true": 5.0, "y_proxy": 4.9, "group": "B"},
            {"y_true": 6.0, "y_proxy": 6.1, "group": "B"},
            {"y_proxy": 5.2, "group": "B"},
            {"y_proxy": 6.1, "group": "B"},
        ]
    )


@pytest.fixture
def estimator() -> StratifiedPPIMeanEstimator:
    return StratifiedPPIMeanEstimator()


# --- _get_strata ---


def test_estimate_raises_when_stratum_has_too_few_labeled(estimator):
    # Drop the second labeled sample in stratum A (line 45 of the doctest), leaving only 1
    records_insufficient_groups = [
        {"y_true": 1.0, "y_proxy": 1.1, "domain": "A"},
        {"y_proxy": 1.8, "domain": "A"},
        {"y_true": 4.0, "y_proxy": 3.9, "domain": "B"},
        {"y_proxy": 4.8, "domain": "B"},
    ]
    dataset = Dataset(records_insufficient_groups)
    with pytest.raises(RuntimeError, match="Too few labeled or unlabeled samples in dataset stratum 'A'"):
        estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", groups_field="domain")


def test_get_strata_splits_correctly(estimator, dataset):
    strata = estimator._get_strata(dataset, "group")
    assert set(strata.keys()) == {"A", "B"}
    assert len(strata["A"]) == 4
    assert len(strata["B"]) == 4
    assert all(r["group"] == "A" for r in strata["A"])
    assert all(r["group"] == "B" for r in strata["B"])


# --- estimate ---


def test_estimate_is_valid_inference_result(estimator, dataset):
    result = estimator.estimate(dataset, y_true_field="y_true", y_proxy_field="y_proxy", groups_field="group")
    assert isinstance(result, SemiSupervisedMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "StratifiedPPIMeanEstimator"


def test_estimate_metadata(estimator, dataset):
    result = estimator.estimate(
        dataset,
        y_true_field="y_true",
        y_proxy_field="y_proxy",
        groups_field="group",
        metric_name="performance",
    )
    assert result.metric_name == "performance"
    assert result.estimator_name == estimator.__class__.__name__
    assert result.n_true == 4
    assert result.n_proxy == 8 == len(dataset)
    assert result.effective_sample_size == 5


def test_estimate_custom_confidence_level(estimator, dataset):
    result = estimator.estimate(
        dataset,
        y_true_field="y_true",
        y_proxy_field="y_proxy",
        groups_field="group",
        confidence_level=0.85,
    )

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


def test_str_format(estimator, dataset):
    result = estimator.estimate(
        dataset,
        y_true_field="y_true",
        y_proxy_field="y_proxy",
        groups_field="group",
        metric_name="performance",
    )
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


def test_repr_equals_str(estimator, dataset):
    result = estimator.estimate(
        dataset,
        y_true_field="y_true",
        y_proxy_field="y_proxy",
        groups_field="group",
        metric_name="perf",
    )
    assert repr(result) == str(result)
