import numpy as np
import pytest
from numpy.typing import NDArray

from glide.core.validation import (
    _validate_budget,
    _validate_equal_lengths,
    _validate_non_constant,
    _validate_pi_consistency,
    _validate_sample_sizes,
    _validate_sampling_probabilities,
    _validate_uncertainties,
    _validate_y_proxy,
    _validate_y_true,
)


def test_validate_y_proxy_valid():
    _validate_y_proxy(np.array([1.0, 2.0]))


def test_validate_y_proxy_nan():
    with pytest.raises(ValueError, match="Input proxy values contain NaN"):
        _validate_y_proxy(np.array([1.0, float("nan")]))


def test_validate_y_proxy_zero_variance():
    with pytest.raises(ValueError, match="Input proxy values have zero variance"):
        _validate_y_proxy(np.array([1.0, 1.0]))


def test_validate_y_proxy_stratum_valid():
    _validate_y_proxy(np.array([1.0, 2.0]), stratum_id="A")


def test_validate_y_proxy_stratum_zero_variance():
    with pytest.raises(ValueError, match="Input proxy values have zero variance in stratum 'A'"):
        _validate_y_proxy(np.array([1.0, 1.0]), stratum_id="A")


def test_validate_y_true_valid():
    _validate_y_true(np.array([1.0, 2.0, np.nan, np.nan]))


def test_validate_y_true_zero_variance():
    with pytest.raises(ValueError, match="Labeled y_true values have zero variance"):
        _validate_y_true(np.array([1.0, 1.0, np.nan]))


def test_validate_uncertainties_valid():
    _validate_uncertainties(np.array([0.1, 0.9]))


def test_validate_uncertainties_nan():
    with pytest.raises(ValueError, match="NaN"):
        _validate_uncertainties(np.array([float("nan"), 0.5]))


def test_validate_uncertainties_non_positive():
    with pytest.raises(ValueError, match="non-positive"):
        _validate_uncertainties(np.array([0.0, 0.5]))


def test_validate_non_constant_valid():
    _validate_non_constant(np.array([1.0, 2.0]))


def test_validate_non_constant_zero_variance():
    with pytest.raises(ValueError, match="rectifiers with zero variance"):
        _validate_non_constant(np.array([1.0, 1.0]))


def test_validate_sampling_probabilities_valid():
    _validate_sampling_probabilities(np.array([0.0, 1.0]))


@pytest.mark.parametrize("bad_pi", [-0.5, 1.5])
def test_validate_sampling_probabilities_out_of_range(bad_pi):
    with pytest.raises(ValueError, match="Sampling probabilities should be in"):
        _validate_sampling_probabilities(np.array([0.5, bad_pi]))


@pytest.fixture
def pi_consistency_mask() -> NDArray:
    return np.array([True, False])


def test_validate_pi_consistency_valid(pi_consistency_mask):
    _validate_pi_consistency(pi_consistency_mask, np.array([0.5, 0.5]))


def test_validate_pi_consistency_labeled_with_zero_pi(pi_consistency_mask):
    with pytest.raises(ValueError, match="non-zero probability of being labeled cannot be labeled"):
        _validate_pi_consistency(pi_consistency_mask, np.array([0.0, 0.5]))


def test_validate_pi_consistency_unlabeled_with_pi_one(pi_consistency_mask):
    with pytest.raises(ValueError, match="probability one of being labeled must be labeled"):
        _validate_pi_consistency(pi_consistency_mask, np.array([0.5, 1.0]))


def test_validate_equal_lengths_valid():
    _validate_equal_lengths(np.array([1.0, 2.0]), np.array([3.0, 4.0]), names=["a", "b"])


def test_validate_equal_lengths_two_arrays():
    with pytest.raises(ValueError, match="y_true and y_proxy must have the same length"):
        _validate_equal_lengths(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0]),
            names=["y_true", "y_proxy"],
        )


def test_validate_equal_lengths_three_arrays():
    with pytest.raises(ValueError, match="y_true, y_proxy, and groups must have the same length"):
        _validate_equal_lengths(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0]),
            names=["y_true", "y_proxy", "groups"],
        )


def test_validate_budget_valid():
    _validate_budget(3, max_size=5)


@pytest.mark.parametrize("bad_budget", [0, -1, 1.5, True])
def test_validate_budget_invalid(bad_budget):
    with pytest.raises(ValueError, match="budget"):
        _validate_budget(bad_budget, max_size=5)


def test_validate_budget_exceeds_max():
    with pytest.raises(ValueError, match="budget"):
        _validate_budget(10, max_size=5)


@pytest.fixture
def too_few_labeled_mask() -> NDArray:
    return np.array([True, False, False])


def test_validate_sample_sizes_valid():
    _validate_sample_sizes(np.array([True, True, False, False]))


def test_validate_sample_sizes_no_stratum_id(too_few_labeled_mask):
    with pytest.raises(ValueError, match="Too few labeled or unlabeled samples in dataset"):
        _validate_sample_sizes(too_few_labeled_mask)


def test_validate_sample_sizes_with_stratum_id(too_few_labeled_mask):
    with pytest.raises(ValueError, match="Too few labeled or unlabeled samples in stratum 'A'"):
        _validate_sample_sizes(too_few_labeled_mask, stratum_id="A")
