import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.inference_result import InferenceResult
from glide.estimators.asi import ASIMeanEstimator


# ── helpers ────────────────────────────────────────────────────────────────────


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
def estimator() -> ASIMeanEstimator:
    return ASIMeanEstimator()


@pytest.fixture
def dataset() -> Dataset:
    return make_dataset(n_labeled=2, n_unlabeled=2)


@pytest.fixture
def simple_y_data(estimator, dataset):
    return estimator._preprocess(dataset, "y_true", "y_proxy", "pi")


# Hand-crafted arrays for deterministic unit tests.
# n=4, n_l=2, pi=0.5 throughout, xi=[1,1,0,0]
HAND_Y_TRUE = np.array([3.0, 5.0, 0.0, 0.0])
HAND_Y_PROXY = np.array([2.0, 4.0, 5.0, 7.0])
HAND_XI = np.array([1.0, 1.0, 0.0, 0.0])
HAND_PI = np.array([0.5, 0.5, 0.5, 0.5])
HAND_Y_DATA = (HAND_Y_TRUE, HAND_Y_PROXY, HAND_XI, HAND_PI)


# ── _preprocess ────────────────────────────────────────────────────────────────


def test_preprocess_returns_four_arrays(estimator, dataset):
    result = estimator._preprocess(dataset, "y_true", "y_proxy", "pi")
    y_true, _, xi, pi = result
    n = len(dataset)
    assert len(result) == 4
    for arr in result:
        assert arr.shape == (n,)
    assert int(xi.sum()) == 2  # n_labeled
    assert not np.any(np.isnan(y_true))
    assert np.all((pi > 0)) and np.all((pi <= 1))


def test_preprocess_xi_is_binary(estimator, dataset):
    _, _, xi, _ = estimator._preprocess(dataset, "y_true", "y_proxy", "pi")
    assert set(xi.tolist()).issubset({0.0, 1.0})


# ── _compute_lambda ────────────────────────────────────────────────────────────


def test_compute_lambda_returns_one_when_power_tuning_false(estimator, simple_y_data):
    lam = estimator._compute_lambda(simple_y_data, power_tuning=False)
    assert lam == 1.0


def test_compute_lambda_known_values(estimator):
    # n=4, n_l=2, pi=0.5
    # a = y_proxy * (xi/pi - 1) = [2.0, 4.0, -3.0, -5.0]
    # b = y_true * xi / pi     = [4.0, 8.0,  0.0,  0.0]
    # cov(b, a, ddof=1) = 46/3,  var(a, ddof=1) = 53/3  =>  lam = 46/53
    y_true = np.array([2.0, 4.0, 0.0, 0.0])
    y_proxy = np.array([2.0, 4.0, 3.0, 5.0])
    xi = np.array([1.0, 1.0, 0.0, 0.0])
    pi = np.array([0.5, 0.5, 0.5, 0.5])
    lam = estimator._compute_lambda((y_true, y_proxy, xi, pi), power_tuning=True)
    assert lam == pytest.approx(46 / 53)


def test_compute_lambda_constant_proxy_returns_zero(estimator):
    # With constant y_proxy = c, a = c*(xi/pi - 1) and
    # cov(b, a) = (c * n_u/n_l) / (n-1) * sum_labeled(y_true).
    # => lambda = 0 exactly when mean_labeled(y_true) = 0.
    # Here y_true_labeled = [-1, 1] has mean 0, so lambda = 0.
    y_true = np.array([-1.0, 1.0, 0.0, 0.0])
    y_proxy = np.array([3.0, 3.0, 3.0, 3.0])
    xi = np.array([1.0, 1.0, 0.0, 0.0])
    pi = np.array([0.5, 0.5, 0.5, 0.5])
    lam = estimator._compute_lambda((y_true, y_proxy, xi, pi), power_tuning=True)
    assert lam == pytest.approx(0.0)


def test_compute_lambda_arbitrary_proxy_known_value(estimator):
    # n=4, n_l=2, pi=0.5
    # a = y_proxy * (xi/pi - 1) = [0, 2,  0, -2], mean(a) = 0
    # b = y_true  * xi / pi     = [2, 6,  0,  0], mean(b) = 2
    # cov(b, a, ddof=1) = 12/3 = 4,  var(a, ddof=1) = 8/3  =>  lam = 3/2
    y_true = np.array([1.0, 3.0, 0.0, 0.0])
    y_proxy = np.array([0.0, 2.0, 0.0, 2.0])
    xi = np.array([1.0, 1.0, 0.0, 0.0])
    pi = np.array([0.5, 0.5, 0.5, 0.5])
    lam = estimator._compute_lambda((y_true, y_proxy, xi, pi), power_tuning=True)
    assert lam == pytest.approx(3 / 2)


def test_compute_lambda_uniform_pi_finite_and_positive(estimator):
    rng = np.random.default_rng(0)
    n, n_l = 20, 5
    pi_val = n_l / n
    y_true_labeled = rng.normal(5.0, 1.0, size=n_l)
    y_proxy_labeled = y_true_labeled + rng.normal(0, 0.3, size=n_l)
    y_proxy_unlabeled = rng.normal(5.0, 1.0, size=n - n_l)
    y_true = np.concatenate([y_true_labeled, np.zeros(n - n_l)])
    y_proxy = np.concatenate([y_proxy_labeled, y_proxy_unlabeled])
    xi = np.array([1.0] * n_l + [0.0] * (n - n_l))
    pi = np.full(n, pi_val)
    lam = estimator._compute_lambda((y_true, y_proxy, xi, pi), power_tuning=True)
    assert np.isfinite(lam)
    assert lam > 0 and lam <= 1


# ── compute_mean_estimate ─────────────────────────────────────────


def test_asi_mean_with_lambda_one_uniform_pi(estimator):
    # lam=1, pi=0.5 (= n_l/n = 2/4)
    # z = [2+2, 4+2, 5, 7] = [4, 6, 5, 7]  =>  mean = 5.5
    # Equivalently: mean(y_proxy_all) + mean_labeled(y_true) - mean_labeled(y_proxy)
    #             = 4.5 + 4.0 - 3.0 = 5.5
    mean = estimator.compute_mean_estimate(HAND_Y_DATA, 1.0)
    expected = np.mean(HAND_Y_PROXY) + np.mean(HAND_Y_TRUE[HAND_XI == 1]) - np.mean(HAND_Y_PROXY[HAND_XI == 1])
    assert mean == pytest.approx(expected)


def test_asi_mean_with_lambda_other(estimator):
    # lam=0.5, pi=0.5
    # z = [0.5*2 + (3-1)/0.5, 0.5*4 + (5-2)/0.5, 0.5*5, 0.5*7]
    #   = [1+4, 2+6, 2.5, 3.5] = [5, 8, 2.5, 3.5]  =>  mean = 19/4 = 4.75
    mean = estimator.compute_mean_estimate(HAND_Y_DATA, 0.5)
    expected = 4.75
    assert mean == pytest.approx(expected)


# ── compute_std_estimate ───────────────────────────────────────────


def test_asi_std_with_lambda_one(estimator):
    # lam=1.0, z=[4, 6, 5, 7], mean=5.5
    # Var(z, ddof=1) = (1.5²+0.5²+0.5²+1.5²)/3 = 5/3  =>  std = sqrt(5/3)/2 = sqrt(5/12)
    std = estimator.compute_std_estimate(HAND_Y_DATA, 1.0)
    expected = np.sqrt(5 / 12)
    assert std == pytest.approx(expected)


def test_asi_std_with_lambda_other(estimator):
    # lam=0.5, z=[5, 8, 2.5, 3.5], mean=4.75
    # diffs=[0.25, 3.25, -2.25, -1.25], sum_sq=17.25
    # Var(z, ddof=1) = 17.25/3 = 5.75  =>  std = sqrt(5.75) / sqrt(4) = sqrt(5.75/4)
    std = estimator.compute_std_estimate(HAND_Y_DATA, 0.5)
    expected = np.sqrt(5.75 / 4)
    assert std == pytest.approx(expected)


# ── estimate ──────────────────────────────────────────────────────────────────


def test_power_tuning_false_is_valid_inference_result(estimator, dataset):
    result = estimator.estimate(
        dataset,
        y_true_field="y_true",
        y_proxy_field="y_proxy",
        sampling_probability_field="pi",
        power_tuning=False,
    )
    assert isinstance(result, InferenceResult)
    assert np.isfinite(result.result.lower_bound)
    assert np.isfinite(result.result.upper_bound)
    assert result.result.lower_bound < result.result.upper_bound
    assert result.estimator_name == "ASIMeanEstimator"


def test_estimate_n_true_and_n_proxy(estimator):
    n_labeled, n_unlabeled = 5, 15
    dataset = make_dataset(n_labeled=n_labeled, n_unlabeled=n_unlabeled)
    result = estimator.estimate(
        dataset,
        y_true_field="y_true",
        y_proxy_field="y_proxy",
        sampling_probability_field="pi",
    )
    assert result.n_true == n_labeled
    assert result.n_proxy == n_labeled + n_unlabeled
