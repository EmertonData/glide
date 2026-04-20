import pytest

from glide.confidence_intervals import CLTConfidenceInterval


def test_default_confidence_level():
    ci = CLTConfidenceInterval(mean=0.0, std=1.0)
    assert ci.confidence_level == 0.95


@pytest.mark.parametrize("confidence_level", [0.0, 1.5])
def test_confidence_level_validation(confidence_level):
    with pytest.raises(ValueError, match="confidence_level must be in \\(0, 1\\)"):
        CLTConfidenceInterval(mean=0.0, std=1.0, confidence_level=confidence_level)


def test_var_computed_from_std():
    ci = CLTConfidenceInterval(mean=0.0, std=2.0)
    assert ci.var == pytest.approx(4.0)


def test_lower_bound():
    ci = CLTConfidenceInterval(mean=0.0, std=1.0, confidence_level=0.95)
    expected = -1.96
    assert ci.lower_bound == pytest.approx(expected, abs=0.0001)


def test_upper_bound():
    ci = CLTConfidenceInterval(mean=0.0, std=1.0, confidence_level=0.95)
    expected = 1.96
    assert ci.upper_bound == pytest.approx(expected, abs=0.0001)


def test_counds_change_with_confidence_level():
    ci = CLTConfidenceInterval(mean=0.0, std=1.0, confidence_level=0.95)
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


def test_null_hypothesis_null_is_true():
    ci = CLTConfidenceInterval(mean=0.0, std=1.0)
    z_stat, p_value, df = ci.test_null_hypothesis(h0_value=0.0)
    assert z_stat == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)
    assert df == float("inf")


def test_null_hypothesis_two_sided():
    ci = CLTConfidenceInterval(mean=1.96, std=1.0)
    z_stat, p_value, df = ci.test_null_hypothesis(h0_value=0.0)
    assert z_stat == pytest.approx(1.96, abs=0.0001)
    assert p_value == pytest.approx(0.05, abs=0.001)
    assert df == float("inf")


def test_null_hypothesis_larger_positive_z():
    ci = CLTConfidenceInterval(mean=1.96, std=1.0)
    z_stat, p_value, df = ci.test_null_hypothesis(h0_value=0.0, alternative="larger")
    assert z_stat == pytest.approx(1.96, abs=0.0001)
    assert p_value == pytest.approx(0.025, abs=0.001)
    assert df == float("inf")


def test_null_hypothesis_larger_negative_z():
    ci = CLTConfidenceInterval(mean=-1.96, std=1.0)
    _, p_value, _ = ci.test_null_hypothesis(h0_value=0.0, alternative="larger")
    assert p_value == pytest.approx(0.975, abs=0.001)


def test_null_hypothesis_smaller_negative_z():
    ci = CLTConfidenceInterval(mean=-1.96, std=1.0)
    z_stat, p_value, df = ci.test_null_hypothesis(h0_value=0.0, alternative="smaller")
    assert z_stat == pytest.approx(-1.96, abs=0.0001)
    assert p_value == pytest.approx(0.025, abs=0.001)
    assert df == float("inf")


def test_null_hypothesis_smaller_positive_z():
    ci = CLTConfidenceInterval(mean=1.96, std=1.0)
    _, p_value, _ = ci.test_null_hypothesis(h0_value=0.0, alternative="smaller")
    assert p_value == pytest.approx(0.975, abs=0.001)


def test_null_hypothesis_invalid_alternative():
    ci = CLTConfidenceInterval(mean=0.0, std=1.0)
    with pytest.raises(ValueError, match="alternative must be 'two-sided', 'larger', or 'smaller'"):
        ci.test_null_hypothesis(h0_value=0.0, alternative="invalid")  # ty: ignore
