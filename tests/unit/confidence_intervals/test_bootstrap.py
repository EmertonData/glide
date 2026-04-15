import numpy as np
import pytest

from glide.confidence_intervals import BootstrapConfidenceInterval


@pytest.fixture
def estimates():
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])


def test_default_confidence_level(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    assert ci.confidence_level == pytest.approx(0.95, abs=0.001)


def test_mean_computed_from_bootstrap_estimates(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    assert ci.mean == pytest.approx(2.25, abs=0.001)


def test_var_computed_from_bootstrap_estimates(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    assert ci.var == pytest.approx(2.292, abs=0.001)


def test_std_computed_from_bootstrap_estimates(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    assert ci.std == pytest.approx(1.514, abs=0.001)


def test_lower_bound(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates, confidence_level=0.95)
    expected_lower = 0.112
    assert ci.lower_bound == pytest.approx(expected_lower, abs=0.001)


def test_upper_bound(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates, confidence_level=0.95)
    expected_upper = 4.388
    assert ci.upper_bound == pytest.approx(expected_upper, abs=0.001)


# --- test_null_hypothesis ---


def test_null_hypothesis_null_is_true():
    estimates = np.array([0.0, 0.0, 0.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    mean, p_value, df = ci.test_null_hypothesis(h0_value=0.0)
    assert mean == pytest.approx(0.0, abs=0.001)
    assert p_value == pytest.approx(1.0, abs=0.001)
    assert df == float("inf")


def test_null_hypothesis_two_sided_some_extreme():
    estimates = np.array([-1.0, 0.0, 1.0, 2.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    mean, p_value, df = ci.test_null_hypothesis(h0_value=0.0)
    assert mean == pytest.approx(0.5, abs=0.001)
    centered = np.abs(estimates - 0.5)  # [1.5, 0.5, 0.5, 1.5]
    observed_deviation = abs(0.5 - 0.0)  # 0.5
    is_extreme = centered >= observed_deviation  # [True, True, True, True]
    expected_p = np.mean(is_extreme)
    assert p_value == pytest.approx(expected_p, abs=0.001)
    assert df == float("inf")


def test_null_hypothesis_larger_positive():
    estimates = np.array([1.0, 2.0, 3.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    mean, p_value, df = ci.test_null_hypothesis(h0_value=0.0, alternative="larger")
    assert mean == pytest.approx(2.0, abs=0.001)
    assert p_value == pytest.approx(0.0, abs=0.001)
    assert df == float("inf")


def test_null_hypothesis_larger_some_support():
    estimates = np.array([-1.0, 0.0, 0.5, 1.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    mean, p_value, df = ci.test_null_hypothesis(h0_value=0.0, alternative="larger")
    assert mean == pytest.approx(0.125, abs=0.001)
    is_at_least_as_extreme = estimates <= 0.0  # [-1.0, 0.0, 0.5, 1.0] <= 0 = [T, T, F, F]
    expected_p = np.mean(is_at_least_as_extreme)
    assert p_value == pytest.approx(expected_p, abs=0.001)
    assert df == float("inf")


def test_null_hypothesis_smaller_some_support():
    estimates = np.array([-1.0, 0.0, 0.5, 1.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    mean, p_value, df = ci.test_null_hypothesis(h0_value=0.0, alternative="smaller")
    assert mean == pytest.approx(0.125, abs=0.001)
    is_at_least_as_extreme = estimates >= 0.0  # [-1.0, 0.0, 0.5, 1.0] >= 0 = [F, T, T, T]
    expected_p = np.mean(is_at_least_as_extreme)
    assert p_value == pytest.approx(expected_p, abs=0.001)
    assert df == float("inf")


def test_null_hypothesis_invalid_alternative():
    estimates = np.array([0.0, 1.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    with pytest.raises(ValueError, match="alternative must be 'two-sided', 'larger', or 'smaller'"):
        ci.test_null_hypothesis(h0_value=0.0, alternative="invalid")
