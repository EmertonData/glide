"""Functional tests for StratifiedPPIMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

from typing import Literal

import numpy as np
import pytest

from glide.estimators.ppi import PPIMeanEstimator
from glide.estimators.stratified_ppi import StratifiedPPIMeanEstimator
from glide.samplers.stratified import StratifiedSampler
from glide.simulators import generate_gaussian_dataset

# ── tests ──────────────────────────────────────────────────────────────────────


def test_two_equal_strata_matches_ppi():
    """Stratified PPI on two identical strata matches PPI on the full doubled dataset.

    When both strata contain identical samples, the stratified estimator reduces to
    PPI++ applied to the combined dataset. Mean and std must match within floating-point
    tolerance since both estimators see the same data distribution at the same scale.
    """
    n_labeled, n_unlabeled = 3, 4

    # Generate base numpy arrays
    y_true, y_proxy = generate_gaussian_dataset(n_labeled=n_labeled, n_unlabeled=n_unlabeled, random_seed=0)

    # Per-stratum PPI reference (single copy)
    ppi_single = PPIMeanEstimator().estimate(y_true, y_proxy)

    # Build stratified arrays: stratum A and B are identical copies

    y_true_stratified = np.hstack([y_true, y_true])
    y_proxy_stratified = np.hstack([y_proxy, y_proxy])
    groups_stratified = np.hstack([np.full(len(y_true), "A"), np.full(len(y_true), "B")])

    result = StratifiedPPIMeanEstimator().estimate(y_true_stratified, y_proxy_stratified, groups_stratified)

    # Mean must match the single-stratum PPI mean (both strata are identical)
    assert result.mean == pytest.approx(ppi_single.mean, abs=1e-10)

    # Std of the stratified estimator must equal single-stratum std / sqrt(2):
    # weighted_var = 0.5^2 * sigma^2 + 0.5^2 * sigma^2 = 0.5 * sigma^2
    assert result.std == pytest.approx(ppi_single.std / np.sqrt(2), abs=1e-10)


def test_stratified_ppi_narrower_ci_with_heterogeneous_strata():
    """Stratified PPI yields a narrower CI than standard PPI on heterogeneous strata.

    When strata differ in proxy quality, per-stratum lambda adaptation reduces the
    total variance compared to a single global lambda estimated on the pooled dataset.
    """
    random_seed = 42
    n_labeled, n_unlabeled = 5, 6

    # Stratum A: low proxy noise
    y_true_a, y_proxy_a = generate_gaussian_dataset(
        n_labeled=n_labeled, n_unlabeled=n_unlabeled, true_mean=0.6, true_std=0.1, random_seed=random_seed
    )
    # Stratum B: high proxy noise → lower lambda is optimal
    y_true_b, y_proxy_b = generate_gaussian_dataset(
        n_labeled=n_labeled, n_unlabeled=n_unlabeled, true_mean=0.4, true_std=1.5, random_seed=random_seed
    )

    # Build stratified arrays
    y_true_stratified = np.hstack([y_true_a, y_true_b])
    y_proxy_stratified = np.hstack([y_proxy_a, y_proxy_b])
    groups_stratified = np.hstack([np.full(len(y_true_a), 0), np.full(len(y_true_b), 1)])

    # Standard PPI on the pooled dataset (ignores group structure)
    y_true_pooled = np.hstack([y_true_a, y_true_b])
    y_proxy_pooled = np.hstack([y_proxy_a, y_proxy_b])
    ppi_result = PPIMeanEstimator().estimate(y_true_pooled, y_proxy_pooled)

    stratified_result = StratifiedPPIMeanEstimator().estimate(y_true_stratified, y_proxy_stratified, groups_stratified)

    # Stratified CI must be strictly narrower
    eps = 1e-1
    assert stratified_result.width < ppi_result.width - eps


def test_neyman_allocation_yields_narrower_ci_than_proportional():
    """Neyman allocation achieves narrower confidence intervals than proportional allocation.

    When strata have heterogeneous variance, Neyman allocation optimally allocates the
    labeling budget by sampling more heavily from high-variance strata. This reduces
    overall variance compared to proportional allocation, which ignores stratum variance.
    """
    random_seed = 0
    budget = 20

    # Common parameters
    n_labeled_per_stratum = 200
    n_unlabeled = 0
    true_mean = 0.5
    proxy_mean = 0.5
    low_std = 0.2
    high_std = 2.0

    # Stratum A: low variance (std_A = 0.2)
    y_true_a, y_proxy_a = generate_gaussian_dataset(
        n_labeled=n_labeled_per_stratum,
        n_unlabeled=n_unlabeled,
        true_mean=true_mean,
        true_std=low_std,
        proxy_mean=proxy_mean,
        proxy_std=low_std,
        random_seed=random_seed,
    )

    # Stratum B: high variance (std_B = 2.0, ratio 10:1)
    y_true_b, y_proxy_b = generate_gaussian_dataset(
        n_labeled=n_labeled_per_stratum,
        n_unlabeled=n_unlabeled,
        true_mean=true_mean,
        true_std=high_std,
        proxy_mean=proxy_mean,
        proxy_std=high_std,
        random_seed=random_seed,
    )

    # Build full population: both strata, all records labeled
    y_true_full = np.hstack([y_true_a, y_true_b])
    y_proxy_full = np.hstack([y_proxy_a, y_proxy_b])
    groups_full = np.hstack([np.full(len(y_true_a), "A"), np.full(len(y_true_b), "B")])

    # Helper to run pipeline for a given strategy
    def pipeline(strategy: Literal["proportional", "neyman"]):
        # Sample using stratified sampler
        _, xi = StratifiedSampler().sample(
            y_proxy_full, groups_full, budget=budget, strategy=strategy, random_seed=random_seed
        )

        # Reconstruct PPI dataset: restore y_true only for sampled records
        sampled_mask = xi == 1
        y_true_ppi = np.full_like(y_true_full, np.nan)
        y_true_ppi[sampled_mask] = y_true_full[sampled_mask]

        # Estimate using stratified PPI
        result = StratifiedPPIMeanEstimator().estimate(y_true_ppi, y_proxy_full, groups_full)
        return result

    neyman_result = pipeline("neyman")
    proportional_result = pipeline("proportional")

    # Neyman allocation must yield a strictly narrower CI
    assert neyman_result.width < proportional_result.width
