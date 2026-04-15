import numpy as np
import pytest

from glide.confidence_intervals import BootstrapConfidenceInterval


def test_default_confidence_level():
    estimates = np.array([0.0, 1.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    assert ci.confidence_level == 0.95


def test_mean_computed_from_bootstrap_estimates():
    estimates = np.array([1.0, 3.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    assert ci.mean == pytest.approx(2.0)


def test_var_computed_from_bootstrap_estimates():
    estimates = np.array([1.0, 5.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    assert ci.var == pytest.approx(8.0)  # sample variance with ddof=1


def test_std_computed_from_bootstrap_estimates():
    estimates = np.array([1.0, 5.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    assert ci.std == pytest.approx(np.sqrt(8.0))


def test_lower_bound():
    rng = np.random.default_rng(0)
    estimates = rng.normal(loc=0.0, scale=1.0, size=100)
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates, confidence_level=0.95)
    expected_lower = np.quantile(estimates, 0.025)
    assert ci.lower_bound == pytest.approx(expected_lower)


def test_upper_bound():
    rng = np.random.default_rng(0)
    estimates = rng.normal(loc=0.0, scale=1.0, size=100)
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates, confidence_level=0.95)
    expected_upper = np.quantile(estimates, 0.975)
    assert ci.upper_bound == pytest.approx(expected_upper)


# --- test_null_hypothesis ---


def test_null_hypothesis_null_is_true():
    estimates = np.array([0.0, 0.0, 0.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    mean, p_value, df = ci.test_null_hypothesis(h0_value=0.0)
    assert mean == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)
    assert df == float("inf")


def test_null_hypothesis_two_sided_some_extreme():
    estimates = np.array([-1.0, 0.0, 1.0, 2.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    mean, p_value, df = ci.test_null_hypothesis(h0_value=0.0)
    assert mean == pytest.approx(0.5)
    centered = np.abs(estimates - 0.5)  # [1.5, 0.5, 0.5, 1.5]
    observed_deviation = abs(0.5 - 0.0)  # 0.5
    is_extreme = centered >= observed_deviation  # [True, True, True, True]
    expected_p = np.mean(is_extreme)
    assert p_value == pytest.approx(expected_p)
    assert df == float("inf")


def test_null_hypothesis_larger_positive():
    estimates = np.array([1.0, 2.0, 3.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    mean, p_value, df = ci.test_null_hypothesis(h0_value=0.0, alternative="larger")
    assert mean == pytest.approx(2.0)
    assert p_value == pytest.approx(0.0)
    assert df == float("inf")


def test_null_hypothesis_larger_some_support():
    estimates = np.array([-1.0, 0.0, 0.5, 1.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    mean, p_value, df = ci.test_null_hypothesis(h0_value=0.0, alternative="larger")
    assert mean == pytest.approx(0.125)
    is_at_least_as_extreme = estimates <= 0.0  # [-1.0, 0.0, 0.5, 1.0] <= 0 = [T, T, F, F]
    expected_p = np.mean(is_at_least_as_extreme)
    assert p_value == pytest.approx(expected_p)
    assert df == float("inf")


def test_null_hypothesis_smaller_some_support():
    estimates = np.array([-1.0, 0.0, 0.5, 1.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    mean, p_value, df = ci.test_null_hypothesis(h0_value=0.0, alternative="smaller")
    assert mean == pytest.approx(0.125)
    is_at_least_as_extreme = estimates >= 0.0  # [-1.0, 0.0, 0.5, 1.0] >= 0 = [F, T, T, T]
    expected_p = np.mean(is_at_least_as_extreme)
    assert p_value == pytest.approx(expected_p)
    assert df == float("inf")


def test_null_hypothesis_invalid_alternative():
    estimates = np.array([0.0, 1.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    with pytest.raises(ValueError, match="alternative must be 'two-sided', 'larger', or 'smaller'"):
        ci.test_null_hypothesis(h0_value=0.0, alternative="invalid")
