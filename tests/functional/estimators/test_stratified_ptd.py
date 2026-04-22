"""Functional tests for StratifiedPTDMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.core.simulated_datasets import generate_gaussian_dataset
from glide.estimators.ptd import PTDMeanEstimator
from glide.estimators.stratified_ptd import StratifiedPTDMeanEstimator

# ── tests ──────────────────────────────────────────────────────────────────────


def test_two_equal_strata_matches_ptd():
    """Stratified PTD on two identical strata matches PTD on the full doubled dataset.

    When both strata contain identical samples, the stratified estimator and the
    pooled PTD estimator see the same data distribution at the same scale, so their
    mean and std must agree within Monte Carlo tolerance.
    """
    n_labeled, n_unlabeled = 5, 8
    random_seed = 0
    n_bootstrap = 2000

    # Generate base numpy arrays
    y_true, y_proxy = generate_gaussian_dataset(n_labeled, n_unlabeled, random_seed=random_seed)

    # Per-stratum PTD reference (single copy)
    result_single = PTDMeanEstimator().estimate(y_true, y_proxy, n_bootstrap=n_bootstrap, random_seed=random_seed)

    # Build stratified arrays: stratum A and B are identical copies
    y_true_stratified = np.hstack([y_true, y_true])
    y_proxy_stratified = np.hstack([y_proxy, y_proxy])
    groups = np.hstack([np.full(len(y_true), "A"), np.full(len(y_true), "B")])

    result_stratified = StratifiedPTDMeanEstimator().estimate(
        y_true_stratified, y_proxy_stratified, groups, n_bootstrap=n_bootstrap, random_seed=random_seed
    )

    # Mean must match the single-stratum PTD mean (both strata are identical)
    assert result_stratified.mean == pytest.approx(result_single.mean, abs=0.01)

    # Std of the stratified estimator must equal single-stratum std / sqrt(2):
    assert result_stratified.std == pytest.approx(result_single.std / np.sqrt(2), abs=0.01)


def test_single_stratum_matches_ptd():
    """Stratified PTD with a single stratum reproduces PTD exactly.

    With one stratum covering all data, the stratified estimator applies the same
    bootstrap procedure as PTD on the same arrays. Given an identical seed and
    n_bootstrap, the RNG draws are identical so results match to floating-point
    precision.
    """
    n_labeled, n_unlabeled = 5, 8
    random_seed = 7
    n_bootstrap = 2000

    y_true, y_proxy = generate_gaussian_dataset(n_labeled, n_unlabeled, random_seed=random_seed)
    groups = np.full(len(y_true), "A")

    stratified_result = StratifiedPTDMeanEstimator().estimate(
        y_true, y_proxy, groups, n_bootstrap=n_bootstrap, random_seed=random_seed
    )
    ptd_result = PTDMeanEstimator().estimate(y_true, y_proxy, n_bootstrap=n_bootstrap, random_seed=random_seed)

    assert stratified_result.mean == pytest.approx(ptd_result.mean, abs=1e-10)
    assert stratified_result.std == pytest.approx(ptd_result.std, abs=1e-10)


def test_stratified_ptd_narrower_ci_with_heterogeneous_strata():
    """Stratified PTD yields a narrower CI than standard PTD on heterogeneous strata.

    When strata differ in proxy quality, per-stratum lambda adaptation reduces the
    total variance compared to a single global lambda estimated on the pooled dataset.
    """
    random_seed = 42
    n_labeled, n_unlabeled = 5, 6

    # Stratum A: low proxy noise
    y_true_a, y_proxy_a = generate_gaussian_dataset(
        n_labeled, n_unlabeled, true_mean=0.6, true_std=0.1, random_seed=random_seed
    )
    # Stratum B: high proxy noise → lower lambda is optimal
    y_true_b, y_proxy_b = generate_gaussian_dataset(
        n_labeled, n_unlabeled, true_mean=0.4, true_std=1.5, random_seed=random_seed
    )

    # Build data arrays
    y_true_pooled = np.hstack([y_true_a, y_true_b])
    y_proxy_pooled = np.hstack([y_proxy_a, y_proxy_b])
    groups = np.hstack([np.full(len(y_true_a), 0), np.full(len(y_true_b), 1)])

    # Standard PTD on the pooled dataset (ignores group structure)
    ptd_result = PTDMeanEstimator().estimate(y_true_pooled, y_proxy_pooled, n_bootstrap=2000, random_seed=random_seed)

    stratified_result = StratifiedPTDMeanEstimator().estimate(
        y_true_pooled, y_proxy_pooled, groups, n_bootstrap=2000, random_seed=random_seed
    )

    # Stratified CI must be strictly narrower
    eps = 1e-1
    assert stratified_result.width < ptd_result.width - eps
