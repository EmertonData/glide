"""Functional tests for StratifiedClassicalMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.estimators.classical import ClassicalMeanEstimator
from glide.estimators.stratified_classical import StratifiedClassicalMeanEstimator
from glide.simulators import generate_gaussian_dataset

# ── tests ──────────────────────────────────────────────────────────────────────


def test_two_equal_strata_matches_classical():
    """Stratified classical on two identical strata produces equivalent mean to classical on a single stratum.

    When both strata contain identical samples, the stratified estimator's mean matches
    the classical estimator applied to a single stratum. The stratified std is the single-stratum std
    divided by sqrt(2), due to the effective sample size doubling when combining two identical strata.
    """
    n_labeled = 4

    y_single, _ = generate_gaussian_dataset(n_labeled=n_labeled, n_unlabeled=0, random_seed=0)

    classical_single = ClassicalMeanEstimator().estimate(y_single)

    y = np.hstack([y_single, y_single])
    groups = np.repeat(["A", "B"], n_labeled)

    result = StratifiedClassicalMeanEstimator().estimate(y, groups)

    # Mean must match the single-stratum classical mean (both strata are identical)
    assert result.mean == pytest.approx(classical_single.mean, abs=1e-10)

    # Std of the stratified estimator must equal single-stratum std / sqrt(2):
    # weighted_var = 0.5^2 * sigma^2 + 0.5^2 * sigma^2 = 0.5 * sigma^2
    assert result.std == pytest.approx(classical_single.std / np.sqrt(2), abs=1e-10)


def test_stratified_classical_narrower_ci_with_heterogeneous_strata():
    """Stratified classical yields a narrower CI than standard classical on heterogeneous strata.

    When strata differ in their means, the pooled classical estimator's variance is
    inflated by the between-strata variance. The stratified estimator avoids this by
    computing per-stratum means independently, yielding a narrower confidence interval.
    """
    random_seed = 42
    n_labeled = 8

    y_a, _ = generate_gaussian_dataset(
        n_labeled=n_labeled,
        n_unlabeled=0,
        true_mean=0.0,
        true_std=0.1,
        random_seed=random_seed,
    )
    y_b, _ = generate_gaussian_dataset(
        n_labeled=n_labeled,
        n_unlabeled=0,
        true_mean=1.0,
        true_std=0.1,
        random_seed=random_seed,
    )

    y = np.hstack([y_a, y_b])
    groups = np.repeat(["A", "B"], n_labeled)

    classical_result = ClassicalMeanEstimator().estimate(y)
    stratified_result = StratifiedClassicalMeanEstimator().estimate(y, groups)

    # Stratified CI must be strictly narrower
    eps = 1e-1
    assert stratified_result.width < classical_result.width - eps
