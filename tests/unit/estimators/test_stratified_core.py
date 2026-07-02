from unittest.mock import patch

import numpy as np
import pytest

import glide.estimators.stratified_core as stratified_core_module
from glide.estimators.stratified_core import _preprocess


@pytest.fixture
def y_true():
    return np.array([5.0, 6.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan])


@pytest.fixture
def y_proxy():
    return np.array([4.9, 6.1, 5.2, 6.1, 4.9, 6.1, 5.2, 6.1])


@pytest.fixture
def groups():
    return np.array(["A", "A", "A", "A", "B", "B", "B", "B"])


# --- _preprocess ---
def test_preprocess_delegates_to_validation(y_true, y_proxy, groups):
    with (
        patch.object(stratified_core_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(stratified_core_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(stratified_core_module, "_validate_y_proxy") as mock_validate_y_proxy,
        patch.object(stratified_core_module, "_split_labeled_unlabeled") as mock_split_labeled_unlabeled,
        patch.object(stratified_core_module, "_validate_sample_sizes") as mock_validate_sample_sizes,
    ):
        mock_split_labeled_unlabeled.return_value = (
            np.array([5.0, 6.0]),
            np.array([4.9, 6.1]),
            np.array([5.2, 6.1]),
            np.array([True, True, False, False]),
        )
        _preprocess(y_true, y_proxy, groups)

        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], groups)
        assert mock_validate_has_no_nan.call_args[0][1] == "groups"

        mock_validate_equal_lengths.assert_called_once_with(
            y_true, y_proxy, groups, names=["y_true", "y_proxy", "groups"]
        )

        assert mock_validate_y_proxy.call_count == 3
        np.testing.assert_array_equal(mock_validate_y_proxy.call_args_list[0][0][0], y_proxy)
        y_proxy_a = y_proxy[groups == "A"]
        np.testing.assert_array_equal(mock_validate_y_proxy.call_args_list[1][0][0], y_proxy_a)
        assert mock_validate_y_proxy.call_args_list[1][0][1] == "A"
        y_proxy_b = y_proxy[groups == "B"]
        np.testing.assert_array_equal(mock_validate_y_proxy.call_args_list[2][0][0], y_proxy_b)
        assert mock_validate_y_proxy.call_args_list[2][0][1] == "B"

        assert mock_split_labeled_unlabeled.call_count == 2
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args_list[0][0][0], y_true[groups == "A"])
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args_list[0][0][1], y_proxy[groups == "A"])
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args_list[1][0][0], y_true[groups == "B"])
        np.testing.assert_array_equal(mock_split_labeled_unlabeled.call_args_list[1][0][1], y_proxy[groups == "B"])

        assert mock_validate_sample_sizes.call_count == 2
        labeled_mask_a = ~np.isnan(y_true[groups == "A"])
        np.testing.assert_array_equal(mock_validate_sample_sizes.call_args_list[0][0][0], labeled_mask_a)
        assert mock_validate_sample_sizes.call_args_list[0][0][1] == "A"
        labeled_mask_b = ~np.isnan(y_true[groups == "B"])
        np.testing.assert_array_equal(mock_validate_sample_sizes.call_args_list[1][0][0], labeled_mask_b)
        assert mock_validate_sample_sizes.call_args_list[1][0][1] == "B"


def test_preprocess_valid_output(y_true, y_proxy, groups):
    strata = _preprocess(y_true, y_proxy, groups)

    assert len(strata) == 2
    for y_true_filtered, y_proxy_labeled, y_proxy_unlabeled in strata:
        np.testing.assert_array_equal(y_true_filtered, np.array([5.0, 6.0]))
        np.testing.assert_array_equal(y_proxy_labeled, np.array([4.9, 6.1]))
        np.testing.assert_array_equal(y_proxy_unlabeled, np.array([5.2, 6.1]))
