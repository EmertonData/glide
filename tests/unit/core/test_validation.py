from typing import List
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from glide.core.validation import (
    _get_non_zero_mask,
    _validate_binary_or_nan,
    _validate_bounds,
    _validate_budget_bound,
    _validate_equal_lengths,
    _validate_has_no_nan,
    _validate_label_prob_consistency,
    _validate_literal,
    _validate_min_samples,
    _validate_non_constant,
    _validate_non_empty,
    _validate_probabilities,
    _validate_sample_sizes,
    _validate_strictly_positive,
    _validate_uncertainties,
    _validate_unique_clusters,
    _validate_y_proxy,
    _validate_y_true,
    _validate_y_true_burn_in,
)

# --- helpers ---


@pytest.fixture
def pi_consistency_mask() -> NDArray:
    return np.array([True, False])


@pytest.fixture
def too_few_labeled_mask() -> NDArray:
    return np.array([True, False, False])


@pytest.fixture
def alternatives() -> List[str]:
    return ["two-sided", "larger", "smaller"]


# --- _validate_non_constant ---


def test_validate_non_constant_valid():
    _validate_non_constant(np.array([1.0, 2.0]), "error")


def test_validate_non_constant_raises():
    with pytest.raises(ValueError, match="custom error"):
        _validate_non_constant(np.array([3.0, 3.0]), "custom error")


# --- _validate_has_no_nan ---


def test_validate_has_no_nan_valid():
    _validate_has_no_nan(np.array([1.0, 2.0]), "x")


def test_validate_has_no_nan_raises():
    with pytest.raises(ValueError, match="'x' contains NaN values"):
        _validate_has_no_nan(np.array([1.0, float("nan")]), "x")


def test_validate_has_no_nan_valid_non_numeric():
    _validate_has_no_nan(np.array(["a", "b"]), "x")


def test_validate_has_no_nan_raises_none_in_non_numeric():
    with pytest.raises(ValueError, match="'x' contains None values"):
        _validate_has_no_nan(np.array(["a", None], dtype=object), "x")


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


def test_validate_y_proxy_delegates():
    with (
        patch("glide.core.validation._validate_has_no_nan") as mock_validate_has_no_nan,
        patch("glide.core.validation._validate_non_constant") as mock_validate_non_constant,
    ):
        arr = np.array([1.0, 2.0])
        _validate_y_proxy(arr)
        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], arr)
        assert mock_validate_has_no_nan.call_args[0][1] == "y_proxy"
        mock_validate_non_constant.assert_called_once()
        np.testing.assert_array_equal(mock_validate_non_constant.call_args[0][0], arr)
        assert mock_validate_non_constant.call_args[0][1] == "'y_proxy' values are constant."


# --- _validate_y_true ---


def test_validate_y_true_valid():
    _validate_y_true(np.array([1.0, 2.0, np.nan, np.nan]))


def test_validate_y_true_only_nans():
    with pytest.raises(ValueError, match="only NaN values"):
        _validate_y_true(np.array([np.nan, np.nan]))


def test_validate_y_true_delegates_to_validate_non_constant():
    with patch("glide.core.validation._validate_non_constant") as mock_validate_non_constant:
        arr = np.array([1.0, 1.0, np.nan])
        _validate_y_true(arr)
        mock_validate_non_constant.assert_called_once()
        labeled = arr[~np.isnan(arr)]
        np.testing.assert_array_equal(mock_validate_non_constant.call_args[0][0], labeled)
        assert mock_validate_non_constant.call_args[0][1] == "'y_true' labeled values are constant."


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
    # necessary for 100% coverage
    with pytest.raises(ValueError, match="'y_true', 'y_proxy', and 'groups' must have the same length"):
        _validate_equal_lengths(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0]),
            names=["y_true", "y_proxy", "groups"],
        )


# --- _validate_y_true_burn_in ---


def test_validate_y_true_burn_in_delegates():
    with (
        patch("glide.core.validation._validate_non_empty") as mock_validate_non_empty,
        patch("glide.core.validation._validate_has_no_nan") as mock_validate_has_no_nan,
        patch("glide.core.validation._validate_non_constant") as mock_validate_non_constant,
    ):
        arr = np.array([1.0, 2.0])
        _validate_y_true_burn_in(arr)
        mock_validate_non_empty.assert_called_once()
        np.testing.assert_array_equal(mock_validate_non_empty.call_args[0][0], arr)
        assert mock_validate_non_empty.call_args[0][1] == "y_true"
        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], arr)
        assert mock_validate_has_no_nan.call_args[0][1] == "y_true"
        mock_validate_non_constant.assert_called_once()
        np.testing.assert_array_equal(mock_validate_non_constant.call_args[0][0], arr)
        assert mock_validate_non_constant.call_args[0][1] == "'y_true' label values are constant."


# --- _validate_non_empty ---


def test_validate_non_empty_valid():
    _validate_non_empty([1, 2], "x")


def test_validate_non_empty_raises():
    with pytest.raises(ValueError, match="'x' must be non-empty"):
        _validate_non_empty([], "x")


# --- _validate_bounds ---


@pytest.mark.parametrize("value", [-1.0, 0.0, 1.0])
def test_validate_bounds_closed_valid(value):
    _validate_bounds(value, "x", lower=-1, upper=1)


@pytest.mark.parametrize("bad_value", [-1.1, 1.1])
def test_validate_bounds_closed_raises(bad_value):
    with pytest.raises(ValueError, match="'x' must be in \\[-1, 1\\]"):
        _validate_bounds(bad_value, "x", lower=-1, upper=1)


@pytest.mark.parametrize("value", [0.001, 0.5, 0.999])
def test_validate_bounds_open_valid(value):
    _validate_bounds(value, "x", lower=0, upper=1, left_inclusive=False, right_inclusive=False)


@pytest.mark.parametrize("bad_value", [-0.1, 0.0, 1.0, 1.1])
def test_validate_bounds_open_raises(bad_value):
    with pytest.raises(ValueError, match="'x' must be in \\(0, 1\\)"):
        _validate_bounds(bad_value, "x", lower=0, upper=1, left_inclusive=False, right_inclusive=False)


def test_validate_bounds_default_infinite():
    _validate_bounds(-1e10, "x")
    _validate_bounds(1e10, "x")


def test_validate_bounds_array_valid():
    _validate_bounds(np.array([0.1, 0.9]), "x", lower=0, upper=1, left_inclusive=False, right_inclusive=False)


def test_validate_bounds_array_raises():
    with pytest.raises(ValueError, match="'x' must be in"):
        _validate_bounds(np.array([0.5, 1.0]), "x", lower=0, upper=1, left_inclusive=False, right_inclusive=False)


def test_validate_bounds_custom_error_message():
    with pytest.raises(ValueError, match="custom error"):
        _validate_bounds(2.0, "x", upper=1, error_message="custom error")


# --- _validate_uncertainties ---


def test_validate_uncertainties_delegates():
    with (
        patch("glide.core.validation._validate_has_no_nan") as mock_validate_has_no_nan,
        patch("glide.core.validation._validate_bounds") as mock_validate_bounds,
    ):
        arr = np.array([0.1, 0.9])
        _validate_uncertainties(arr)
        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], arr)
        assert mock_validate_has_no_nan.call_args[0][1] == "uncertainties"
        mock_validate_bounds.assert_called_once()
        np.testing.assert_array_equal(mock_validate_bounds.call_args[0][0], arr)
        assert mock_validate_bounds.call_args[0][1] == "uncertainties"
        assert mock_validate_bounds.call_args[1]["lower"] == 0
        assert mock_validate_bounds.call_args[1]["left_inclusive"] is False
        expected_msg = "'uncertainties' must all be strictly positive; got a non-positive value."
        assert mock_validate_bounds.call_args[1]["error_message"] == expected_msg


# --- _validate_strictly_positive ---


def test_validate_strictly_positive_delegates():
    with patch("glide.core.validation._validate_bounds") as mock_validate_bounds:
        _validate_strictly_positive(1.0, "x")
        mock_validate_bounds.assert_called_once_with(
            1.0, "x", lower=0, left_inclusive=False, error_message="'x' must be strictly positive; got 1.0."
        )


# --- _validate_probabilities ---


def test_validate_probabilities_delegates():
    with patch("glide.core.validation._validate_bounds") as mock_validate_bounds:
        arr = np.array([0.0, 1.0])
        _validate_probabilities(arr)
        mock_validate_bounds.assert_called_once()
        np.testing.assert_array_equal(mock_validate_bounds.call_args[0][0], arr)
        assert mock_validate_bounds.call_args[0][1] == "Probabilities"
        assert mock_validate_bounds.call_args[1]["lower"] == 0
        assert mock_validate_bounds.call_args[1]["upper"] == 1
        assert mock_validate_bounds.call_args[1]["error_message"] == "Probabilities must be in [0, 1]."


# --- _validate_budget_bound ---


def test_validate_budget_bound_delegates():
    with patch("glide.core.validation._validate_bounds") as mock_validate_bounds:
        _validate_budget_bound(3, n_max=5)
        expected_msg = "'budget' must not exceed the number of samples; got budget=3 but the dataset has 5 elements."
        mock_validate_bounds.assert_called_once_with(3, "budget", upper=5, error_message=expected_msg)


# --- _validate_literal ---


@pytest.mark.parametrize("alternative", ["two-sided", "larger", "smaller"])
def test_validate_literal_valid(alternative, alternatives):
    _validate_literal(alternative, "param", alternatives)


def test_validate_literal_raises(alternatives):
    with pytest.raises(ValueError, match="'param' must be"):
        _validate_literal("something", "param", alternatives)


# --- _validate_sample_sizes ---


def test_validate_sample_sizes_valid():
    _validate_sample_sizes(np.array([True, True, False, False]))


def test_validate_sample_sizes_no_stratum_id(too_few_labeled_mask):
    with pytest.raises(ValueError, match="Too few labeled or unlabeled samples in dataset"):
        _validate_sample_sizes(too_few_labeled_mask)


def test_validate_sample_sizes_with_stratum_id(too_few_labeled_mask):
    with pytest.raises(ValueError, match="Too few labeled or unlabeled samples in stratum 'A'"):
        _validate_sample_sizes(too_few_labeled_mask, stratum_id="A")


# --- _validate_binary_or_nan ---


@pytest.mark.parametrize("valid", [np.array([0, 1, np.nan]), np.array([0.0, 1.0])])
def test_validate_binary_or_nan_valid(valid):
    _validate_binary_or_nan(valid, "x")


def test_validate_binary_or_nan_raises():
    with pytest.raises(ValueError, match="'x' must only contain 0, 1, and np.nan"):
        _validate_binary_or_nan(np.array([0.5, 1.0]), "x")


# --- _validate_min_samples ---


def test_validate_min_samples_valid():
    _validate_min_samples(np.array([1.0, 2.0]), "y")


def test_validate_min_samples_too_few():
    with pytest.raises(ValueError, match="'y' must have at least 2 non-NaN values; got 1"):
        _validate_min_samples(np.array([1.0]), "y")


def test_validate_min_samples_too_few_with_stratum():
    with pytest.raises(ValueError, match="per stratum; got 1 in stratum 'A'"):
        _validate_min_samples(np.array([1.0]), "y", stratum_id="A")


# --- _validate_unique_clusters ---


def test_validate_unique_clusters_valid():
    _validate_unique_clusters(np.array(["A", "B"]), np.array(["C", "D"]))


def test_validate_unique_clusters_raises_on_intersection():
    with pytest.raises(ValueError, match="Cluster 'A' contains both labeled and unlabeled observations."):
        _validate_unique_clusters(np.array(["A", "B"]), np.array(["A", "C"]))
