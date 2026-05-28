import numpy as np
import pytest
from numpy.typing import NDArray

from glide.core.validation import (
    _get_non_zero_mask,
    _is_constant,
    _validate_budget_bound,
    _validate_burn_in_y_true,
    _validate_equal_lengths,
    _validate_has_no_nan,
    _validate_label_prob_consistency,
    _validate_probabilities,
    _validate_sample_sizes,
    _validate_strictly_positive,
    _validate_uncertainties,
    _validate_y_proxy,
    _validate_y_true,
)

# --- helpers ---


@pytest.fixture
def pi_consistency_mask() -> NDArray:
    return np.array([True, False])


@pytest.fixture
def too_few_labeled_mask() -> NDArray:
    return np.array([True, False, False])


# --- _is_constant ---


def test_is_constant_true():
    assert _is_constant(np.array([3.0, 3.0]))


def test_is_constant_false():
    assert not _is_constant(np.array([1.0, 2.0]))


# --- _validate_has_no_nan ---


def test_validate_has_no_nan_valid():
    _validate_has_no_nan(np.array([1.0, 2.0]), "x")


def test_validate_has_no_nan_raises():
    with pytest.raises(ValueError, match="'x' contains NaN values"):
        _validate_has_no_nan(np.array([1.0, float("nan")]), "x")


# --- _get_non_zero_mask ---


def test_get_non_zero_mask_all_positive():
    result = _get_non_zero_mask(np.array([0.5, 1.0]))
    np.testing.assert_array_equal(result, np.array([True, True]))


def test_get_non_zero_mask_with_zero_emits_warning():
    expected = np.array([False, True])
    warning_message = "some warning"
    with pytest.warns(UserWarning, match=warning_message):
        result = _get_non_zero_mask(np.array([0.0, 0.5]), warning_message=warning_message)
    np.testing.assert_array_equal(result, expected)


# --- _validate_y_proxy ---


def test_validate_y_proxy_valid():
    _validate_y_proxy(np.array([1.0, 2.0]))


def test_validate_y_proxy_nan():
    with pytest.raises(ValueError, match="'y_proxy' contains NaN"):
        _validate_y_proxy(np.array([1.0, float("nan")]))


def test_validate_y_proxy_constant():
    with pytest.raises(ValueError, match="'y_proxy' values are constant"):
        _validate_y_proxy(np.array([1.0, 1.0]))


def test_validate_y_proxy_stratum_valid():
    _validate_y_proxy(np.array([1.0, 2.0]), stratum_id="A")


def test_validate_y_proxy_stratum_constant():
    with pytest.raises(ValueError, match="'y_proxy' values are constant in stratum 'A'"):
        _validate_y_proxy(np.array([1.0, 1.0]), stratum_id="A")


# --- _validate_y_true ---


def test_validate_y_true_valid():
    _validate_y_true(np.array([1.0, 2.0, np.nan, np.nan]))


def test_validate_y_true_only_nans():
    with pytest.raises(ValueError, match="only NaN values"):
        _validate_y_true(np.array([np.nan, np.nan]))


def test_validate_y_true_constant():
    with pytest.raises(ValueError, match="'y_true' labeled values are constant"):
        _validate_y_true(np.array([1.0, 1.0, np.nan]))


# --- _validate_uncertainties ---


def test_validate_uncertainties_valid():
    _validate_uncertainties(np.array([0.1, 0.9]))


def test_validate_uncertainties_nan():
    with pytest.raises(ValueError, match="NaN"):
        _validate_uncertainties(np.array([float("nan"), 0.5]))


def test_validate_uncertainties_non_positive():
    with pytest.raises(ValueError, match="non-positive"):
        _validate_uncertainties(np.array([0.0, 0.5]))


# --- _validate_probabilities ---


def test_validate_probabilities_valid():
    _validate_probabilities(np.array([0.0, 1.0]))


@pytest.mark.parametrize("bad_pi", [-0.5, 1.5])
def test_validate_probabilities_out_of_range(bad_pi):
    with pytest.raises(ValueError, match="Sampling probabilities must be in"):
        _validate_probabilities(np.array([0.5, bad_pi]))


# --- _validate_label_prob_consistency ---


def test_validate_label_prob_consistency_valid(pi_consistency_mask):
    _validate_label_prob_consistency(pi_consistency_mask, np.array([0.5, 0.5]))


def test_validate_label_prob_consistency_labeled_with_zero_pi(pi_consistency_mask):
    with pytest.raises(ValueError, match="zero probability of being labeled cannot be labeled"):
        _validate_label_prob_consistency(pi_consistency_mask, np.array([0.0, 0.5]))


def test_validate_label_prob_consistency_unlabeled_with_pi_one(pi_consistency_mask):
    with pytest.raises(ValueError, match="probability one of being labeled must be labeled"):
        _validate_label_prob_consistency(pi_consistency_mask, np.array([0.5, 1.0]))


# --- _validate_equal_lengths ---


def test_validate_equal_lengths_valid():
    _validate_equal_lengths(np.array([1.0, 2.0]), np.array([3.0, 4.0]), names=["a", "b"])


def test_validate_equal_lengths_two_arrays():
    with pytest.raises(ValueError, match="'y_true' and 'y_proxy' must have the same length"):
        _validate_equal_lengths(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0]),
            names=["y_true", "y_proxy"],
        )


def test_validate_equal_lengths_three_arrays():
    with pytest.raises(ValueError, match="'y_true', 'y_proxy', and 'groups' must have the same length"):
        _validate_equal_lengths(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0]),
            names=["y_true", "y_proxy", "groups"],
        )


# --- _validate_burn_in_y_true ---


def test_validate_burn_in_y_true_valid():
    _validate_burn_in_y_true(np.array([1.0, 2.0]))


def test_validate_burn_in_y_true_empty():
    with pytest.raises(ValueError, match="non-empty"):
        _validate_burn_in_y_true(np.array([]))


def test_validate_burn_in_y_true_nan():
    with pytest.raises(ValueError, match="NaN"):
        _validate_burn_in_y_true(np.array([1.0, float("nan")]))


def test_validate_burn_in_y_true_constant():
    with pytest.raises(ValueError, match="label values are constant"):
        _validate_burn_in_y_true(np.array([1.0, 1.0]))


# --- _validate_strictly_positive ---


def test_validate_strictly_positive_valid():
    _validate_strictly_positive(1.0, "x")


@pytest.mark.parametrize("bad_value", [0.0, -1.0])
def test_validate_strictly_positive_invalid(bad_value):
    with pytest.raises(ValueError, match="must be strictly positive"):
        _validate_strictly_positive(bad_value, "x")


# --- _validate_budget_bound ---


def test_validate_budget_bound_valid():
    _validate_budget_bound(3, n_max=5)


def test_validate_budget_bound_exceeds_max():
    with pytest.raises(ValueError, match="budget"):
        _validate_budget_bound(10, n_max=5)


# --- _validate_sample_sizes ---


def test_validate_sample_sizes_valid():
    _validate_sample_sizes(np.array([True, True, False, False]))


def test_validate_sample_sizes_no_stratum_id(too_few_labeled_mask):
    with pytest.raises(ValueError, match="Too few labeled or unlabeled samples in dataset"):
        _validate_sample_sizes(too_few_labeled_mask)


def test_validate_sample_sizes_with_stratum_id(too_few_labeled_mask):
    with pytest.raises(ValueError, match="Too few labeled or unlabeled samples in stratum 'A'"):
        _validate_sample_sizes(too_few_labeled_mask, stratum_id="A")
