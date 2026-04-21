import numpy as np
import pytest

from glide.simulators.stratified_binary import generate_stratified_binary_dataset


def test_generate_stratified_binary_dataset_structure_and_counts():
    y_true, y_proxy, groups = generate_stratified_binary_dataset(
        n=[1, 2],
        N=[2, 1],
        true_mean=[0.6, 0.8],
        proxy_mean=[0.5, 0.7],
        correlation=[0.75, 0.75],
        random_seed=0,
    )
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_proxy, np.ndarray)
    assert isinstance(groups, np.ndarray)
    assert len(y_true) == 6
    assert len(y_proxy) == 6
    assert len(groups) == 6
    assert np.sum(~np.isnan(y_true)) == 3  # 1 + 2 labeled samples
    assert np.sum(~np.isnan(y_proxy)) == 6  # all proxy samples present
    np.testing.assert_array_equal(groups, [0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(y_true, np.array([1.0, np.nan, np.nan, 1.0, 1.0, np.nan]), equal_nan=True)
    np.testing.assert_allclose(y_proxy, np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]))


def test_generate_stratified_binary_dataset_empty_strata_raises():
    with pytest.raises(ValueError, match=r"Number of strata must be at least 1, got 0"):
        generate_stratified_binary_dataset(n=[], N=[], true_mean=[], proxy_mean=[], correlation=[])


def test_generate_stratified_binary_dataset_mismatched_lists_raises():
    with pytest.raises(ValueError, match=r"All input lists must have the same length"):
        generate_stratified_binary_dataset(
            n=[1, 2],
            N=[2, 1, 3],
            true_mean=[0.5, 0.6],
            proxy_mean=[0.5, 0.6],
            correlation=[0.8, 0.8],
        )


def test_generate_stratified_binary_dataset_reproducibility():
    y_true1, y_proxy1, groups1 = generate_stratified_binary_dataset(
        n=[1, 2],
        N=[2, 1],
        true_mean=[0.6, 0.8],
        proxy_mean=[0.5, 0.7],
        correlation=[0.75, 0.75],
        random_seed=42,
    )
    y_true2, y_proxy2, groups2 = generate_stratified_binary_dataset(
        n=[1, 2],
        N=[2, 1],
        true_mean=[0.6, 0.8],
        proxy_mean=[0.5, 0.7],
        correlation=[0.75, 0.75],
        random_seed=42,
    )
    np.testing.assert_allclose(y_true1, y_true2, equal_nan=True)
    np.testing.assert_allclose(y_proxy1, y_proxy2, equal_nan=True)
    np.testing.assert_array_equal(groups1, groups2)
