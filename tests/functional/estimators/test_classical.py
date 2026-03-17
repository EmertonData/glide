"""Functional tests for ClassicalMeanEstimator.

These tests verify end-to-end behaviour: given a realistic dataset, the
estimator produces a valid confidence interval that covers the true mean at
the nominal rate across Monte Carlo repetitions.
"""

import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.inference_result import ClassicalMeanInferenceResult
from glide.estimators.classical import ClassicalMeanEstimator

TRUE_MEAN = 0.6
N = 200
CONFIDENCE_LEVEL = 0.95
N_SEEDS = 500


def make_dataset(n: int, true_mean: float, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, true_mean, size=n).astype(float)
    return Dataset([{"y_true": float(v)} for v in y])


# --- basic sanity ---


def test_returns_classical_inference_result():
    dataset = make_dataset(n=N, true_mean=TRUE_MEAN, seed=0)
    result = ClassicalMeanEstimator().estimate(dataset, y_field="y_true")
    assert isinstance(result, ClassicalMeanInferenceResult)


def test_n_equals_dataset_size():
    dataset = make_dataset(n=N, true_mean=TRUE_MEAN, seed=0)
    result = ClassicalMeanEstimator().estimate(dataset, y_field="y_true")
    assert result.n == N


def test_ci_bounds_are_finite():
    dataset = make_dataset(n=N, true_mean=TRUE_MEAN, seed=0)
    result = ClassicalMeanEstimator().estimate(dataset, y_field="y_true")
    assert np.isfinite(result.confidence_interval.lower_bound)
    assert np.isfinite(result.confidence_interval.upper_bound)
    assert result.confidence_interval.lower_bound < result.confidence_interval.upper_bound


def test_width_property():
    dataset = make_dataset(n=N, true_mean=TRUE_MEAN, seed=0)
    result = ClassicalMeanEstimator().estimate(dataset, y_field="y_true")
    expected_width = result.confidence_interval.upper_bound - result.confidence_interval.lower_bound
    assert result.width == pytest.approx(expected_width)


# --- empirical coverage ---


def test_empirical_coverage_at_nominal_level():
    """CI should cover the true mean approximately CONFIDENCE_LEVEL of the time."""
    estimator = ClassicalMeanEstimator()
    hits = 0
    for seed in range(N_SEEDS):
        ds = make_dataset(n=N, true_mean=TRUE_MEAN, seed=seed)
        result = estimator.estimate(ds, y_field="y_true", confidence_level=CONFIDENCE_LEVEL)
        if result.confidence_interval.lower_bound <= TRUE_MEAN <= result.confidence_interval.upper_bound:
            hits += 1
    observed_coverage = hits / N_SEEDS
    # Allow ±3 percentage points tolerance
    assert abs(observed_coverage - CONFIDENCE_LEVEL) < 0.03
