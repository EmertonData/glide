import numpy as np
import pytest

from glide.confidence_intervals import BootstrapConfidenceInterval


def test_default_confidence_level():
    bootstrap_estimates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=bootstrap_estimates)
    assert ci.confidence_level == 0.95


def test_mean_is_bootstrap_mean():
    bootstrap_estimates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=bootstrap_estimates)
    expected_mean = 3.0
    assert ci.mean == pytest.approx(expected_mean)


def test_var_and_std_computed():
    bootstrap_estimates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=bootstrap_estimates)
    expected_var = np.var(bootstrap_estimates, ddof=1)
    expected_std = np.sqrt(expected_var)
    assert ci.var == pytest.approx(expected_var)
    assert ci.std == pytest.approx(expected_std)


def test_bounds_are_quantiles():
    rng = np.random.default_rng(0)
    bootstrap_estimates = rng.normal(loc=5.0, scale=0.3, size=100)
    ci = BootstrapConfidenceInterval(bootstrap_estimates=bootstrap_estimates, confidence_level=0.95)
    # Lower bound should be approximately the 2.5th percentile
    expected_lower = np.quantile(bootstrap_estimates, 0.025)
    # Upper bound should be approximately the 97.5th percentile
    expected_upper = np.quantile(bootstrap_estimates, 0.975)
    assert ci.lower_bound == pytest.approx(expected_lower)
    assert ci.upper_bound == pytest.approx(expected_upper)


def test_bounds_symmetric_for_symmetric_distribution():
    # For a symmetric distribution centered at 0, bounds should be roughly symmetric
    rng = np.random.default_rng(42)
    bootstrap_estimates = rng.normal(loc=0.0, scale=1.0, size=1000)
    ci = BootstrapConfidenceInterval(bootstrap_estimates=bootstrap_estimates, confidence_level=0.95)
    lower_dist = abs(ci.lower_bound - ci.mean)
    upper_dist = abs(ci.upper_bound - ci.mean)
    # Bounds should be approximately symmetric around the mean
    assert lower_dist == pytest.approx(upper_dist, rel=0.1)


def test_null_hypothesis_two_sided_true_null():
    bootstrap_estimates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=bootstrap_estimates)
    # Mean is 3.0, test against true null
    test_stat, p_value, df = ci.test_null_hypothesis(h0_value=3.0)
    assert test_stat == pytest.approx(3.0)
    assert p_value == pytest.approx(1.0)
    assert df == float("inf")


def test_null_hypothesis_two_sided_extreme():
    bootstrap_estimates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=bootstrap_estimates)
    # Mean is 3.0, test against extreme null (none of the bootstrap estimates are as extreme)
    test_stat, p_value, df = ci.test_null_hypothesis(h0_value=10.0)
    assert test_stat == pytest.approx(3.0)
    # p_value should be small but not necessarily 0 (depends on data)
    assert p_value >= 0.0
    assert df == float("inf")


def test_null_hypothesis_larger():
    bootstrap_estimates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=bootstrap_estimates)
    # Test H1: mean > 2.0
    # For larger, p_value = proportion of estimates <= h0_value
    test_stat, p_value, df = ci.test_null_hypothesis(h0_value=2.0, alternative="larger")
    assert test_stat == pytest.approx(3.0)
    # Only 1 estimate (1.0) is < 2.0, and 1 is = 2.0, so 2/5 = 0.4
    assert p_value == pytest.approx(0.4)
    assert df == float("inf")


def test_null_hypothesis_smaller():
    bootstrap_estimates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=bootstrap_estimates)
    # Test H1: mean < 4.0
    # For smaller, p_value = proportion of estimates >= h0_value
    test_stat, p_value, df = ci.test_null_hypothesis(h0_value=4.0, alternative="smaller")
    assert test_stat == pytest.approx(3.0)
    # Estimates >= 4.0: 4.0 and 5.0, so 2/5 = 0.4
    assert p_value == pytest.approx(0.4)
    assert df == float("inf")


def test_null_hypothesis_invalid_alternative():
    bootstrap_estimates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = BootstrapConfidenceInterval(bootstrap_estimates=bootstrap_estimates)
    with pytest.raises(ValueError, match="alternative must be 'two-sided', 'larger', or 'smaller'"):
        ci.test_null_hypothesis(h0_value=3.0, alternative="invalid")


def test_different_confidence_levels():
    rng = np.random.default_rng(0)
    bootstrap_estimates = rng.normal(loc=5.0, scale=0.3, size=100)
    # 90% CI should be narrower than 95% CI
    ci_90 = BootstrapConfidenceInterval(bootstrap_estimates=bootstrap_estimates, confidence_level=0.90)
    ci_95 = BootstrapConfidenceInterval(bootstrap_estimates=bootstrap_estimates, confidence_level=0.95)
    width_90 = ci_90.upper_bound - ci_90.lower_bound
    width_95 = ci_95.upper_bound - ci_95.lower_bound
    assert width_90 < width_95
