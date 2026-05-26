import numpy as np
import pytest

from glide.estimators import ClassicalMeanEstimator
from glide.scientific_validation import compute_hits, coverage_with_error_bar, run_monte_carlo


@pytest.fixture
def confidence_levels():
    return np.array([0.9])


@pytest.fixture
def run_seed():
    def _generate(seed):
        y = np.array([0.0, 1.0])
        result = ClassicalMeanEstimator().estimate(y, confidence_level=0.9)
        return {
            "M": {
                "mean": result.mean,
                "std": result.std,
                "confidence_interval": result.confidence_interval,
                "effective_sample_size": 2,
            }
        }

    return _generate


@pytest.fixture
def stats():
    return {
        "M": {
            "lower_bounds": {0.9: np.array([-0.3224, -0.3224])},
            "upper_bounds": {0.9: np.array([1.3224, 1.3224])},
        }
    }


@pytest.fixture
def hits():
    return np.array([1.0, 1.0, 0.0, 1.0])


# --- run_monte_carlo ---


@pytest.mark.parametrize("n_seeds", [-1, 0])
def test_run_monte_carlo_raises_on_invalid_n_seeds(confidence_levels, run_seed, n_seeds):
    with pytest.raises(ValueError, match="n_seeds"):
        run_monte_carlo(confidence_levels, run_seed, n_seeds=n_seeds)


def test_run_monte_carlo_raises_on_empty_run_seed(confidence_levels):
    with pytest.raises(ValueError, match="run_seed"):
        run_monte_carlo(confidence_levels, lambda seed: {}, n_seeds=2)


@pytest.mark.parametrize("invalid_confidence_level", [-0.5, 0, 1, 2])
def test_run_monte_carlo_raises_on_invalid_confidence_level(run_seed, invalid_confidence_level):
    with pytest.raises(ValueError, match="confidence_levels"):
        run_monte_carlo(np.array([0.2, 0.5, 0.8, invalid_confidence_level]), run_seed, n_seeds=2)


def test_run_monte_carlo_output(confidence_levels, run_seed):
    level = confidence_levels[0]
    result = run_monte_carlo(confidence_levels, run_seed, n_seeds=2)
    expected = {
        "means": np.array([0.5, 0.5]),
        "stds": np.array([0.5, 0.5]),
        "lower_bounds": {level: np.full(2, -0.322427)},
        "upper_bounds": {level: np.full(2, 1.322427)},
        "effective_sample_sizes": np.array([2.0, 2.0]),
    }
    np.testing.assert_allclose(result["M"]["means"], expected["means"])
    np.testing.assert_allclose(result["M"]["stds"], expected["stds"])
    np.testing.assert_allclose(result["M"]["lower_bounds"][level], expected["lower_bounds"][level], atol=1e-6)
    np.testing.assert_allclose(result["M"]["upper_bounds"][level], expected["upper_bounds"][level], atol=1e-6)
    np.testing.assert_allclose(result["M"]["effective_sample_sizes"], expected["effective_sample_sizes"])


# --- compute_hits ---


@pytest.mark.parametrize("invalid_confidence_level", [-1, 0, 1, 2])
def test_compute_hits_raises_on_invalid_confidence_level(stats, invalid_confidence_level):
    with pytest.raises(ValueError, match="confidence_level"):
        compute_hits(stats, confidence_level=invalid_confidence_level, true_mean=0.5)


def test_compute_hits_raises_on_missing_confidence_level(stats):
    # 0.8 was not passed to run_monte_carlo (fixture uses [0.9]), so it is absent from stats
    with pytest.raises(ValueError, match="confidence_level"):
        compute_hits(stats, confidence_level=0.8, true_mean=0.5)


def test_compute_hits(stats):
    hits = compute_hits(stats, confidence_level=0.9, true_mean=0.5)
    np.testing.assert_array_equal(hits["M"], np.ones(2))


# --- coverage_with_error_bar ---


def test_coverage_with_error_bar_raises_on_empty_hits():
    with pytest.raises(ValueError, match="hits"):
        coverage_with_error_bar(np.array([]), confidence_level=0.95)


@pytest.mark.parametrize("invalid_confidence_level", [-0.5, 0, 1, 2])
def test_coverage_with_error_bar_raises_on_invalid_confidence_level(invalid_confidence_level):
    with pytest.raises(ValueError, match="confidence_level"):
        coverage_with_error_bar(np.array([1.0, 0.0]), confidence_level=invalid_confidence_level)


def test_coverage_with_error_bar_expected_values(hits):
    mean_cov, lower, upper = coverage_with_error_bar(hits, confidence_level=0.95)
    assert mean_cov == pytest.approx(0.75, abs=1e-10)
    assert lower == pytest.approx(0.26, abs=0.01)
    assert upper == pytest.approx(1.24, abs=0.01)
