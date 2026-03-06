import pytest

from glide.core.clt_confidence_interval import CLTConfidenceInterval


def test_default_confidence_level():
    ci = CLTConfidenceInterval(mean=0.0, std=1.0)
    assert ci.confidence_level == 0.95


def test_lower_bound():
    ci = CLTConfidenceInterval(mean=0.0, std=1.0, confidence_level=0.95)
    expected = -1.96
    assert ci.lower_bound == pytest.approx(expected, abs=0.0001)


def test_upper_bound():
    ci = CLTConfidenceInterval(mean=0.0, std=1.0, confidence_level=0.95)
    expected = 1.96
    assert ci.upper_bound == pytest.approx(expected, abs=0.0001)
