import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.mean_inference_result import SemiSupervisedMeanInferenceResult
from glide.estimators.asi import ASIMeanEstimator

# --- helpers ---


def make_dataset(n_labeled: int = 2, n_unlabeled: int = 2, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    pi = n_labeled / (n_labeled + n_unlabeled)
    y_true_vals = rng.normal(4.0, 1.0, size=n_labeled)
    y_proxy_labeled = y_true_vals + rng.normal(0, 0.2, size=n_labeled)
    y_proxy_unlabeled = rng.normal(4.0, 1.0, size=n_unlabeled)
    labeled = [{"y_true": float(yt), "y_proxy": float(yp), "pi": pi} for yt, yp in zip(y_true_vals, y_proxy_labeled)]
    unlabeled = [{"y_proxy": float(yp), "pi": pi} for yp in y_proxy_unlabeled]
    return Dataset(labeled + unlabeled)


@pytest.fixture
def dataset() -> Dataset:
    return make_dataset(n_labeled=2, n_unlabeled=2)


@pytest.fixture
def estimator() -> ASIMeanEstimator:
    return ASIMeanEstimator()


@pytest.fixture
def simple_y_data(estimator, dataset):
    return estimator._preprocess(dataset, "y_true", "y_proxy", "pi")


# Hand-crafted arrays for deterministic unit tests.
Y_TRUE = np.array([3.0, 5.0, 0.0, 0.0])
Y_PROXY = np.array([2.0, 4.0, 5.0, 7.0])
XI = np.array([1.0, 1.0, 0.0, 0.0])
PI = np.array([0.5, 0.5, 0.5, 0.5])
Y_DATA = (Y_TRUE, Y_PROXY, XI, PI)


# --- preprocessing ---


def test_preprocess_counts(estimator, dataset):
    y_true, y_proxy, xi, pi = estimator._preprocess(dataset, "y_true", "y_proxy", "pi")
    assert len(y_true) == 4
    assert len(y_proxy) == 4
    assert len(xi) == 4
    assert int(xi.sum()) == 2
    assert len(pi) == 4


def test_preprocess_no_nans_in_y_true(estimator, dataset):
    y_true, _, _, _ = estimator._preprocess(dataset, "y_true", "y_proxy", "pi")
    assert not np.any(np.isnan(y_true))


def test_preprocess_returns_four_arrays(estimator, dataset):
    result = estimator._preprocess(dataset, "y_true", "y_proxy", "pi")
    n = len(dataset)
    assert len(result) == 4
    for arr in result:
        assert arr.shape == (n,)


def test_preprocess_pi_values_in_valid_range(estimator, dataset):
    _, _, _, pi = estimator._preprocess(dataset, "y_true", "y_proxy", "pi")
    assert np.all((pi > 0)) and np.all((pi <= 1))


def test_preprocess_xi_is_binary(estimator, dataset):
    _, _, xi, _ = estimator._preprocess(dataset, "y_true", "y_proxy", "pi")
    assert set(xi.tolist()).issubset({0.0, 1.0})


@pytest.mark.parametrize("bad_pi", [0.0, -0.5])
def test_preprocess_raises_on_non_positive_pi(estimator, bad_pi):
    labeled = [{"y_true": 1.0, "y_proxy": 1.0, "pi": 0.5}]
    unlabeled = [{"y_proxy": 1.0, "pi": bad_pi}]
    dataset = Dataset(labeled + unlabeled)
    with pytest.raises(AssertionError, match="Minimum annotation probability is <= 0"):
        estimator._preprocess(dataset, "y_true", "y_proxy", "pi")


# --- _compute_lambda ---


def test_compute_lambda_returns_one_when_power_tuning_false(estimator, simple_y_data):
    lam = estimator._compute_lambda(simple_y_data, power_tuning=False)
    assert lam == 1.0


def test_compute_lambda_known_values(estimator):
    y_true = np.array([2.0, 4.0, 0.0, 0.0])
    y_proxy = np.array([2.0, 4.0, 3.0, 5.0])
    xi = np.array([1.0, 1.0, 0.0, 0.0])
    pi = np.array([0.5, 0.5, 0.5, 0.5])
    lam = estimator._compute_lambda((y_true, y_proxy, xi, pi), power_tuning=True)
    assert lam == pytest.approx(46 / 53)


def test_compute_lambda_constant_proxy_returns_zero(estimator):
    y_true = np.array([-1.0, 1.0, 0.0, 0.0])
    y_proxy = np.array([3.0, 3.0, 3.0, 3.0])
    xi = np.array([1.0, 1.0, 0.0, 0.0])
    pi = np.array([0.5, 0.5, 0.5, 0.5])
    lam = estimator._compute_lambda((y_true, y_proxy, xi, pi), power_tuning=True)
    assert lam == pytest.approx(0.0)


# --- _compute_mean_estimate ---


def test_asi_mean_with_lambda_other(estimator):
    _lambda = 0.5
    rectified_labels = _lambda * Y_PROXY + XI * (Y_TRUE - _lambda * Y_PROXY) / PI
    mean = estimator._compute_mean_estimate(rectified_labels)
    expected = 4.75
    assert mean == pytest.approx(expected)


# --- _compute_std_estimate ---


def test_asi_std_with_lambda_other(estimator):
    # lam=0.5, z=[5, 8, 2.5, 3.5], mean=4.75
    # diffs=[0.25, 3.25, -2.25, -1.25], sum_sq=17.25
    # Var(z, ddof=1) = 17.25/3 = 5.75  =>  std = sqrt(5.75) / sqrt(4) = sqrt(5.75/4)
    _lambda = 0.5
    rectified_labels = _lambda * Y_PROXY + XI * (Y_TRUE - _lambda * Y_PROXY) / PI
    std = estimator._compute_std_estimate(rectified_labels)
    expected = np.sqrt(5.75 / 4)
    assert std == pytest.approx(expected)


# --- estimate ---


def test_power_tuning_false_is_valid_inference_result(estimator, dataset):
    result = estimator.estimate(
        dataset,
        y_true_field="y_true",
        y_proxy_field="y_proxy",
        sampling_probability_field="pi",
        power_tuning=False,
    )
    assert isinstance(result, SemiSupervisedMeanInferenceResult)
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound
    assert result.estimator_name == "ASIMeanEstimator"



def test_estimate_full_output(estimator):
    # Build a dataset from the Y_DATA hand-crafted constants (n=4, n_l=2, pi=0.5).
    labeled = [
        {"y_true": float(Y_TRUE[0]), "y_proxy": float(Y_PROXY[0]), "pi": float(PI[0])},
        {"y_true": float(Y_TRUE[1]), "y_proxy": float(Y_PROXY[1]), "pi": float(PI[1])},
    ]
    unlabeled = [
        {"y_proxy": float(Y_PROXY[2]), "pi": float(PI[2])},
        {"y_proxy": float(Y_PROXY[3]), "pi": float(PI[3])},
    ]
    dataset = Dataset(labeled + unlabeled)

    result = estimator.estimate(
        dataset,
        y_true_field="y_true",
        y_proxy_field="y_proxy",
        sampling_probability_field="pi",
        metric_name="TestMetric",
        confidence_level=0.95,
        power_tuning=True,
    )

    expected_mean = 5.341176470
    expected_std = 0.5807365168135927
    expected_lower = 4.20295381
    expected_upper = 6.47939912
    expected_ess = 5

    assert result.metric_name == "TestMetric"
    assert result.estimator_name == "ASIMeanEstimator"
    assert result.n_true == 2
    assert result.n_proxy == 4
    assert result.effective_sample_size == expected_ess
    assert result.confidence_interval.confidence_level == 0.95
    assert result.confidence_interval.mean == pytest.approx(expected_mean)
    assert result.std == pytest.approx(expected_std)
    assert result.confidence_interval.lower_bound == pytest.approx(expected_lower)
    assert result.confidence_interval.upper_bound == pytest.approx(expected_upper)


# --- __str__ / __repr__ ---


def test_str_format(estimator, dataset):
    result = estimator.estimate(
        dataset,
        y_true_field="y_true",
        y_proxy_field="y_proxy",
        sampling_probability_field="pi",
        metric_name="accuracy",
    )
    output = str(result)
    assert "Metric: accuracy" in output
    assert "Point Estimate:" in output
    assert "Confidence Interval (95%):" in output
    assert f"Estimator : {estimator.__class__.__name__}" in output
    assert "n_true: 2" in output
    assert "n_proxy: 4" in output
    assert "Effective Sample Size:" in output


def test_repr_equals_str(estimator, dataset):
    result = estimator.estimate(
        dataset,
        y_true_field="y_true",
        y_proxy_field="y_proxy",
        sampling_probability_field="pi",
        metric_name="perf",
    )
    assert repr(result) == str(result)
