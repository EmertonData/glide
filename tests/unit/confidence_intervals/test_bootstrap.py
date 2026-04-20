import numpy as np
import pytest

from glide.confidence_intervals import BootstrapConfidenceInterval


@pytest.fixture
def estimates():
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])


def test_default_confidence_level(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    assert ci.confidence_level == pytest.approx(0.95, abs=0.001)


@pytest.mark.parametrize("confidence_level", [0.0, 1.5])
def test_confidence_level_validation(estimates, confidence_level):
    with pytest.raises(ValueError, match="confidence_level must be in \\(0, 1\\)"):
        BootstrapConfidenceInterval(bootstrap_estimates=estimates, confidence_level=confidence_level)


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


def test_bounds_change_with_confidence_level(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates, confidence_level=0.95)
    lower_95 = ci.lower_bound
    upper_95 = ci.upper_bound
    width_95 = ci.width

    # Change confidence level
    ci.confidence_level = 0.99
    lower_99 = ci.lower_bound
    upper_99 = ci.upper_bound
    width_99 = ci.width

    # Bounds should be wider at 99% confidence
    assert lower_99 < lower_95
    assert upper_99 > upper_95
    assert width_99 > width_95


# --- test_null_hypothesis ---


def test_null_hypothesis_null_is_true(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    _, p_value, df = ci.test_null_hypothesis(h0_value=2.25)
    assert p_value == pytest.approx(1.0, abs=0.001)
    assert df == float("inf")


def test_null_hypothesis_two_sided(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    _, p_value, df = ci.test_null_hypothesis(h0_value=0.0)
    expected_p = 0.2
    assert p_value == pytest.approx(expected_p, abs=0.001)
    assert df == float("inf")


def test_null_hypothesis_larger(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    _, p_value, df = ci.test_null_hypothesis(h0_value=1.0, alternative="larger")
    expected_p = 0.3
    assert p_value == pytest.approx(expected_p, abs=0.001)
    assert df == float("inf")


def test_null_hypothesis_smaller(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    _, p_value, df = ci.test_null_hypothesis(h0_value=3.0, alternative="smaller")
    expected_p = 0.4
    assert p_value == pytest.approx(expected_p, abs=0.001)
    assert df == float("inf")


def test_null_hypothesis_invalid_alternative(estimates):
    ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates)
    with pytest.raises(ValueError, match="alternative must be 'two-sided', 'larger', or 'smaller'"):
        ci.test_null_hypothesis(h0_value=0.0, alternative="invalid")  # ty: ignore
