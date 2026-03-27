"""Functional tests for StratifiedPPIMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.simulated_datasets import generate_gaussian_dataset
from glide.estimators.ppi import PPIMeanEstimator
from glide.estimators.stratified_ppi import StratifiedPPIMeanEstimator

# ── tests ──────────────────────────────────────────────────────────────────────


def test_two_equal_strata_matches_ppi():
    """Stratified PPI on two identical strata matches PPI on the full doubled dataset.

    When both strata contain identical records, the stratified estimator reduces to
    PPI++ applied to the combined dataset. Mean and std must match within floating-point
    tolerance since both estimators see the same data distribution at the same scale.
    """
    n_labeled, n_unlabeled = 3, 4

    # Build base records (no group)
    single_labeled, single_unlabeled = generate_gaussian_dataset(n_labeled, n_unlabeled, random_seed=0)

    # Per-stratum PPI reference (single copy)
    single_dataset = Dataset(single_labeled + single_unlabeled)
    ppi_single = PPIMeanEstimator().estimate(single_dataset, y_true_field="y_true", y_proxy_field="y_proxy")

    # Stratified dataset: stratum A and B are identical copies of the base data
    records_a = [
        {"y_true": record["y_true"], "y_proxy": record["y_proxy"], "group": "A"}
        for record in single_dataset[:n_labeled]
    ]
    records_a += [{"y_proxy": record["y_proxy"], "group": "A"} for record in single_dataset[n_labeled:]]
    records_b = [
        {"y_true": record["y_true"], "y_proxy": record["y_proxy"], "group": "B"}
        for record in single_dataset[:n_labeled]
    ]
    records_b += [{"y_proxy": record["y_proxy"], "group": "B"} for record in single_dataset[n_labeled:]]
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
    random_seed = 42
    n_labeled, n_unlabeled = 5, 6

    # Stratum A: low proxy noise → high lambda should be beneficial
    labeled_records_a, unlabeled_records_a = generate_gaussian_dataset(
        n_labeled, n_unlabeled, true_mean=0.6, true_std=0.1, random_seed=random_seed
    )
    records_a = labeled_records_a + unlabeled_records_a
    for i in range(len(records_a)):
        records_a[i]["group"] = "A"
    # Stratum B: high proxy noise → lower lambda is optimal
    labeled_records_b, unlabeled_records_b = generate_gaussian_dataset(
        n_labeled, n_unlabeled, true_mean=0.4, true_std=1.5, random_seed=random_seed
    )
    records_b = labeled_records_b + unlabeled_records_b
    for i in range(len(records_b)):
        records_b[i]["group"] = "B"

    stratified_dataset = Dataset(records_a + records_b)

    # Standard PPI on the pooled dataset (ignores group structure)
    pooled_records = [{k: v for k, v in r.items() if k != "group"} for r in stratified_dataset]
    pooled_dataset = Dataset(pooled_records)

    ppi_result = PPIMeanEstimator().estimate(pooled_dataset, y_true_field="y_true", y_proxy_field="y_proxy")
    stratified_result = StratifiedPPIMeanEstimator().estimate(
        stratified_dataset, y_true_field="y_true", y_proxy_field="y_proxy", groups_field="group"
    )

    # Stratified CI must be strictly narrower
    eps = 1e-1
    assert stratified_result.width < ppi_result.width - eps
