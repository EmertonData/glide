"""Functional tests for StratifiedClassicalMeanEstimator.

These tests verify end-to-end statistical properties rather than implementation
details, and therefore require larger datasets to hold reliably.
"""

import numpy as np
import pytest

from glide.core.dataset import Dataset
from glide.core.simulated_datasets import generate_gaussian_dataset
from glide.estimators.classical import ClassicalMeanEstimator
from glide.estimators.stratified_classical import StratifiedClassicalMeanEstimator

# ── tests ──────────────────────────────────────────────────────────────────────


def test_two_equal_strata_matches_classical():
    """Stratified classical on two identical strata matches classical on the full doubled dataset.

    When both strata contain identical records, the stratified estimator reduces to
    the classical mean applied to the combined dataset. Mean and std must match within
    floating-point tolerance since both estimators see the same data distribution at
    the same scale.
    """
    n_labeled = 4

    single_labeled_dataset, _ = generate_gaussian_dataset(n_labeled, 0, random_seed=0)

    classical_single = ClassicalMeanEstimator().estimate(single_labeled_dataset, y_field="y_true")

    y_values = single_labeled_dataset["y_true"]
    y = np.concatenate([y_values, y_values])
    groups = np.array(["A"] * len(y_values) + ["B"] * len(y_values))

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

    labeled_a, _ = generate_gaussian_dataset(n_labeled, 0, true_mean=0.0, true_std=0.1, random_seed=random_seed)
    labeled_b, _ = generate_gaussian_dataset(n_labeled, 0, true_mean=1.0, true_std=0.1, random_seed=random_seed)

    y_a, y_b = labeled_a["y_true"], labeled_b["y_true"]

    y_stratified = np.concatenate([y_a, y_b])
    groups = np.array(["A"] * len(y_a) + ["B"] * len(y_b))
    dataset_obj = Dataset([{"y": val} for val in y_stratified])

    classical_result = ClassicalMeanEstimator().estimate(dataset_obj, y_field="y")
    stratified_result = StratifiedClassicalMeanEstimator().estimate(y_stratified, groups)

    # Stratified CI must be strictly narrower
    eps = 1e-1
    assert stratified_result.width < classical_result.width - eps
