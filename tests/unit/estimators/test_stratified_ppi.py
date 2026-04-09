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


@pytest.fixture
def y_data():
    y_true = np.array([5.0, 6.0, 7.0])
    y_proxy_labeled = np.array([4.5, 5.5, 6.5])
    y_proxy_unlabeled = np.array([6.0, 7.0, 8.0])
    return y_true, y_proxy_labeled, y_proxy_unlabeled


# --- _preprocess ---


def test_preprocess_returns_correct_shapes(estimator, dataset):
    y_true_all, y_proxy_all, groups = estimator._preprocess(dataset, "y_true", "y_proxy", "group")
    assert len(y_true_all) == 8
    assert len(y_proxy_all) == 8
    assert len(groups) == 8
    assert np.sum(~np.isnan(y_true_all)) == 4


def test_preprocess_raises_on_nan_proxy(estimator):
    dataset = Dataset(
        [
            {"y_true": 1.0, "y_proxy": float("nan"), "group": "A"},
            {"y_true": 2.0, "y_proxy": 2.2, "group": "A"},
            {"y_proxy": 1.5, "group": "A"},
            {"y_proxy": 1.8, "group": "A"},
        ]
    )
    with pytest.raises(ValueError, match="Input proxy values contain NaN"):
        estimator._preprocess(dataset, "y_true", "y_proxy", "group")


def test_preprocess_raises_on_zero_variance_proxy_in_stratum(estimator):
    dataset = Dataset(
        [
            {"y_true": 1.0, "y_proxy": 1.0, "group": "A"},
            {"y_true": 2.0, "y_proxy": 1.0, "group": "A"},
            {"y_proxy": 1.0, "group": "A"},
            {"y_proxy": 1.0, "group": "A"},
        ]
    )
    with pytest.raises(ValueError, match="Input proxy values have zero variance in stratum 'A'"):
        estimator._preprocess(dataset, "y_true", "y_proxy", "group")


def test__preprocess_raises_when_stratum_has_too_few_labeled(estimator):
    # Drop the second labeled sample in stratum A (line 45 of the doctest), leaving only 1
    records_insufficient_groups = [
        {"y_true": 1.0, "y_proxy": 1.1, "domain": "A"},
        {"y_proxy": 1.8, "domain": "A"},
        {"y_true": 4.0, "y_proxy": 3.9, "domain": "B"},
        {"y_proxy": 4.8, "domain": "B"},
    ]
    dataset = Dataset(records_insufficient_groups)
    with pytest.raises(RuntimeError, match="Too few labeled or unlabeled samples in dataset stratum 'A'"):
        estimator._preprocess(dataset, y_true_field="y_true", y_proxy_field="y_proxy", groups_field="domain")


def test_preprocess_groups_preserved(estimator, dataset):
    _, _, groups = estimator._preprocess(dataset, "y_true", "y_proxy", "group")
    expected = np.array(["A", "A", "A", "A", "B", "B", "B", "B"], dtype=object)
    assert np.array_equal(groups, expected)


def test_preprocess_raises_on_unknown_y_proxy_field(estimator, dataset):
    with pytest.raises(ValueError):
        estimator._preprocess(dataset, "y_true", "nonexistent_field", "group")


def test_preprocess_raises_on_unknown_groups_field(estimator, dataset):
    with pytest.raises((ValueError, KeyError)):
        estimator._preprocess(dataset, "y_true", "y_proxy", "nonexistent_group_field")


def test_preprocess_raises_on_all_strata_zero_variance(estimator):
    dataset = Dataset(
        [
            {"y_true": 1.0, "y_proxy": 0.0, "group": "A"},
            {"y_true": 2.0, "y_proxy": 0.0, "group": "A"},
            {"y_proxy": 0.0, "group": "A"},
            {"y_proxy": 0.0, "group": "A"},
            {"y_true": 3.0, "y_proxy": 1.0, "group": "B"},
            {"y_true": 4.0, "y_proxy": 1.0, "group": "B"},
            {"y_proxy": 1.0, "group": "B"},
            {"y_proxy": 1.0, "group": "B"},
        ]
    )
    with pytest.raises(ValueError, match="zero variance in stratum"):
        estimator._preprocess(dataset, "y_true", "y_proxy", "group")


# --- _compute_lambda ---


def test_compute_lambda_returns_one_when_power_tuning_false(estimator):
    y_true = np.array([5.0, 6.0])
    y_proxy_labeled = np.array([4.9, 6.1])
    y_proxy_unlabeled = np.array([5.2, 6.1])
    result = estimator._compute_lambda(y_true, y_proxy_labeled, y_proxy_unlabeled, power_tuning=False)
    assert result == 1.0


def test_compute_lambda_known_values(estimator, y_data):
    y_true, y_proxy_labeled, y_proxy_unlabeled = y_data
    expected = 0.34
    result = estimator._compute_lambda(y_true, y_proxy_labeled, y_proxy_unlabeled, power_tuning=True)
    assert result == pytest.approx(expected, abs=0.01)


# --- _compute_mean_estimate ---


def test_compute_mean_estimate_known_values(estimator, y_data):
    y_true, y_proxy_labeled, y_proxy_unlabeled = y_data
    expected = 6.75
    result = estimator._compute_mean_estimate(y_true, y_proxy_labeled, y_proxy_unlabeled, _lambda=0.5)
    assert result == pytest.approx(expected)


# --- _compute_std_estimate ---


def test_compute_std_estimate_known_values(estimator, y_data):
    y_true, y_proxy_labeled, y_proxy_unlabeled = y_data
    expected = 0.41
    result = estimator._compute_std_estimate(y_true, y_proxy_labeled, y_proxy_unlabeled, _lambda=0.5)
    assert result == pytest.approx(expected, abs=1e-2)


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
