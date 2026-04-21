"""Functional tests for StratifiedPTDMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np

from glide.core.simulated_datasets import generate_gaussian_dataset
from glide.estimators.ptd import PTDMeanEstimator
from glide.estimators.stratified_ptd import StratifiedPTDMeanEstimator

# ── tests ──────────────────────────────────────────────────────────────────────


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
