"""Functional tests for StratifiedPPIMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.estimators.ppi import PPIMeanEstimator
from glide.estimators.stratified_ppi import StratifiedPPIMeanEstimator


# ── helpers ────────────────────────────────────────────────────────────────────


def _make_stratum(rng, n_labeled: int, n_unlabeled: int, true_mean: float, noise_std: float, group: str) -> list:
    """Generate records for one stratum with Gaussian true labels and additive proxy noise."""
    y_true = rng.normal(true_mean, 1.0, n_labeled)
    y_proxy_labeled = y_true + rng.normal(0, noise_std, n_labeled)
    y_proxy_unlabeled = rng.normal(true_mean, 1.0, n_unlabeled) + rng.normal(0, noise_std, n_unlabeled)
    labeled = [{"y_true": float(y), "y_proxy": float(yh), "group": group} for y, yh in zip(y_true, y_proxy_labeled)]
    unlabeled = [{"y_proxy": float(yh), "group": group} for yh in y_proxy_unlabeled]
    return labeled + unlabeled


# ── tests ──────────────────────────────────────────────────────────────────────


def test_two_equal_strata_matches_ppi():
    """Stratified PPI on two identical strata matches PPI on the full doubled dataset.

    When both strata contain identical records, the stratified estimator reduces to
    PPI++ applied to the combined dataset. Mean and std must match within floating-point
    tolerance since both estimators see the same data distribution at the same scale.
    """
    rng = np.random.default_rng(0)
    n_labeled, n_unlabeled = 3, 4

    # Build base records (no group)
    y_true = rng.normal(5.0, 1.0, n_labeled)
    y_proxy_labeled = y_true + rng.normal(0, 0.3, n_labeled)
    y_proxy_unlabeled = rng.normal(5.0, 1.0, n_unlabeled) + rng.normal(0, 0.3, n_unlabeled)

    # Per-stratum PPI reference (single copy)
    single_labeled = [{"y_true": float(y), "y_proxy": float(yh)} for y, yh in zip(y_true, y_proxy_labeled)]
    single_unlabeled = [{"y_proxy": float(yh)} for yh in y_proxy_unlabeled]
    single_dataset = Dataset(single_labeled + single_unlabeled)
    ppi_single = PPIMeanEstimator().estimate(single_dataset, y_true_field="y_true", y_proxy_field="y_proxy")

    # Stratified dataset: stratum A and B are identical copies of the base data
    records_a = [{"y_true": float(y), "y_proxy": float(yh), "group": "A"} for y, yh in zip(y_true, y_proxy_labeled)]
    records_a += [{"y_proxy": float(yh), "group": "A"} for yh in y_proxy_unlabeled]
    records_b = [{"y_true": float(y), "y_proxy": float(yh), "group": "B"} for y, yh in zip(y_true, y_proxy_labeled)]
    records_b += [{"y_proxy": float(yh), "group": "B"} for yh in y_proxy_unlabeled]
    stratified_dataset = Dataset(records_a + records_b)

    result = StratifiedPPIMeanEstimator().estimate(
        stratified_dataset, y_true_field="y_true", y_proxy_field="y_proxy", groups_field="group"
    )

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
    rng = np.random.default_rng(42)
    n_labeled, n_unlabeled = 5, 6

    # Stratum A: low proxy noise → high lambda should be beneficial
    records_a = _make_stratum(rng, n_labeled, n_unlabeled, true_mean=0.6, noise_std=0.1, group="A")
    # Stratum B: high proxy noise → lower lambda is optimal
    records_b = _make_stratum(rng, n_labeled, n_unlabeled, true_mean=0.4, noise_std=1.5, group="B")

    stratified_dataset = Dataset(records_a + records_b)

    # Standard PPI on the pooled dataset (ignores group structure)
    pooled_records = [{k: v for k, v in r.items() if k != "group"} for r in stratified_dataset]
    pooled_dataset = Dataset(pooled_records)

    ppi_result = PPIMeanEstimator().estimate(pooled_dataset, y_true_field="y_true", y_proxy_field="y_proxy")
    stratified_result = StratifiedPPIMeanEstimator().estimate(
        stratified_dataset, y_true_field="y_true", y_proxy_field="y_proxy", groups_field="group"
    )

    # Stratified CI must be strictly narrower
    assert stratified_result.width < ppi_result.width
