from unittest.mock import patch

import numpy as np

import glide.estimators.stratified_core as stratified_core_module
from glide.estimators.stratified_core import _preprocess

# --- _preprocess ---


def test_preprocess_returns_correct_shapes():
    y_true = np.array([5.0, 6.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan])
    y_proxy = np.array([4.9, 6.1, 5.2, 6.1, 4.9, 6.1, 5.2, 6.1])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    strata = _preprocess(y_true, y_proxy, groups)
    assert len(strata) == 2
    for y_true_filtered, y_proxy_labeled, y_proxy_unlabeled in strata:
        assert len(y_true_filtered) == 2
        assert len(y_proxy_labeled) == 2
        assert len(y_proxy_unlabeled) == 2


def test_preprocess_delegates_to_validation():
    y_true = np.array([5.0, 6.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan])
    y_proxy = np.array([4.9, 6.1, 5.2, 6.1, 4.9, 6.1, 5.2, 6.1])
    groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    with (
        patch.object(stratified_core_module, "_validate_equal_lengths") as mock_equal_lengths,
        patch.object(stratified_core_module, "_validate_y_proxy") as mock_y_proxy,
        patch.object(stratified_core_module, "_validate_y_true") as mock_y_true,
        patch.object(stratified_core_module, "_validate_sample_sizes") as mock_sample_sizes,
    ):
        _preprocess(y_true, y_proxy, groups)

        mock_equal_lengths.assert_called_once_with(y_true, y_proxy, groups, names=["y_true", "y_proxy", "groups"])
        mock_y_true.assert_called_once_with(y_true)
        assert mock_y_proxy.call_count == 3
        mock_y_proxy.assert_any_call(y_proxy)
        assert mock_sample_sizes.call_count == 2
