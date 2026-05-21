import numpy as np
import pytest

from glide.estimators import ClassicalMeanEstimator
from glide.scientific_validation import compute_hits, coverage_with_errbar, run_monte_carlo


@pytest.fixture
def methods():
    return ["M"]


@pytest.fixture
def confidence_levels():
    return np.array([0.9])


@pytest.fixture
def generate_estimates():
    def _generate(seed):
        y = np.array([0.0, 1.0])
        result = ClassicalMeanEstimator().estimate(y, confidence_level=0.9)
        return {"M": {"mean": result.mean, "std": result.std, "confidence_interval": result.confidence_interval}}

    return _generate


@pytest.fixture
def stats(methods, confidence_levels, generate_estimates):
    return run_monte_carlo(methods, confidence_levels, generate_estimates, n_seeds=2)


# --- run_monte_carlo ---


def test_run_monte_carlo_output_keys(stats):
    assert set(stats["M"].keys()) == {"means", "stds", "lower_bounds", "upper_bounds", "effective_sample_size"}


def test_run_monte_carlo_output_shapes(stats, confidence_levels):
    assert stats["M"]["means"].shape == (2,)
    assert stats["M"]["stds"].shape == (2,)
    assert stats["M"]["lower_bounds"][confidence_levels[0]].shape == (2,)
    assert stats["M"]["upper_bounds"][confidence_levels[0]].shape == (2,)
    assert stats["M"]["effective_sample_size"].shape == (2,)


def test_run_monte_carlo_effective_sample_size_nan_when_absent(stats):
    assert np.all(np.isnan(stats["M"]["effective_sample_size"]))


def test_run_monte_carlo_effective_sample_size_collected_when_present(methods, confidence_levels):
    def generate_estimates(seed):
        y = np.array([0.0, 1.0])
        result = ClassicalMeanEstimator().estimate(y, confidence_level=0.9)
        return {
            "M": {
                "mean": result.mean,
                "std": result.std,
                "confidence_interval": result.confidence_interval,
                "effective_sample_size": 42.0,
            }
        }

    result = run_monte_carlo(methods, confidence_levels, generate_estimates, n_seeds=2)
    np.testing.assert_array_equal(result["M"]["effective_sample_size"], np.array([42.0, 42.0]))


def test_run_monte_carlo_mean_values(stats):
    np.testing.assert_allclose(stats["M"]["means"], [0.5, 0.5])


def test_run_monte_carlo_bounds_ordered(stats, confidence_levels):
    lower = stats["M"]["lower_bounds"][confidence_levels[0]]
    upper = stats["M"]["upper_bounds"][confidence_levels[0]]
    assert np.all(lower < upper)


# --- compute_hits ---


def test_compute_hits_shape(stats, methods):
    hits = compute_hits(stats, confidence_level=0.9, true_mean=0.5, methods=methods)
    assert hits["M"].shape == (2,)


def test_compute_hits_all_hits_when_true_mean_at_center(stats, methods):
    hits = compute_hits(stats, confidence_level=0.9, true_mean=0.5, methods=methods)
    np.testing.assert_array_equal(hits["M"], [1.0, 1.0])


def test_compute_hits_no_hits_when_true_mean_far_outside(stats, methods):
    hits = compute_hits(stats, confidence_level=0.9, true_mean=100.0, methods=methods)
    np.testing.assert_array_equal(hits["M"], [0.0, 0.0])


def test_compute_hits_values_are_binary(stats, methods):
    hits = compute_hits(stats, confidence_level=0.9, true_mean=0.5, methods=methods)
    assert set(hits["M"].tolist()).issubset({0.0, 1.0})


# --- coverage_with_errbar ---


def test_coverage_with_errbar_returns_three_floats():
    hits = np.array([1.0, 1.0, 0.0, 1.0])
    result = coverage_with_errbar(hits, confidence_level=0.95)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_coverage_with_errbar_mean():
    hits = np.array([1.0, 0.0, 1.0, 0.0])
    mean_cov, _, _ = coverage_with_errbar(hits, confidence_level=0.95)
    assert mean_cov == pytest.approx(0.5, abs=1e-10)


def test_coverage_with_errbar_bounds_ordered():
    hits = np.array([1.0, 1.0, 0.0, 1.0])
    mean_cov, lower, upper = coverage_with_errbar(hits, confidence_level=0.95)
    assert lower < mean_cov < upper


def test_coverage_with_errbar_expected_values():
    hits = np.array([1.0, 1.0, 0.0, 1.0])
    mean_cov, lower, upper = coverage_with_errbar(hits, confidence_level=0.95)
    assert mean_cov == pytest.approx(0.75, abs=1e-10)
    assert lower == pytest.approx(0.26, abs=0.01)
    assert upper == pytest.approx(1.24, abs=0.01)
