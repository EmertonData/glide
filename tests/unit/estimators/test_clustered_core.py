from unittest.mock import call, patch

import numpy as np
import pytest
from numpy.typing import NDArray

import glide.estimators.clustered_core as clustered_core_module
from glide.estimators.clustered_core import _preprocess


@pytest.fixture
def y_true() -> NDArray:
    return np.array([4.0, np.nan, 6.0, np.nan])


@pytest.fixture
def y_proxy() -> NDArray:
    return np.array([2.0, 4.0, 6.0, 6.0])


@pytest.fixture
def clusters() -> NDArray:
    return np.array(["A", "B", "C", "D"])


# --- _preprocess ---


def test_preprocess_delegates_to_validation(y_true, y_proxy, clusters):
    with (
        patch.object(clustered_core_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(clustered_core_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(clustered_core_module, "_validate_unique_clusters") as mock_validate_unique_clusters,
        patch.object(clustered_core_module, "_validate_bounds") as mock_validate_bounds,
    ):
        _preprocess(y_true, y_proxy, clusters)

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y_true)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], y_proxy)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][2], clusters)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y_true", "y_proxy", "clusters"]}

        assert len(mock_validate_has_no_nan.call_args_list) == 2
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args_list[0][0][0], y_proxy)
        assert mock_validate_has_no_nan.call_args_list[0][0][1] == "y_proxy"
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args_list[1][0][0], clusters)
        assert mock_validate_has_no_nan.call_args_list[1][0][1] == "clusters"

        mock_validate_unique_clusters.assert_called_once()
        np.testing.assert_array_equal(mock_validate_unique_clusters.call_args[0][0], np.array(["A", "C"]))
        np.testing.assert_array_equal(mock_validate_unique_clusters.call_args[0][1], np.array(["B", "D"]))

        mock_validate_bounds.assert_has_calls(
            [
                call(2, "n_labeled_clusters", lower=2, error_message="Need at least 2 fully labeled clusters; got 2."),
                call(
                    2, "n_unlabeled_clusters", lower=2, error_message="Need at least 2 fully unlabeled clusters; got 2."
                ),
            ]
        )


def test_preprocess_valid_output(y_true, y_proxy, clusters):
    labeled_true_means, labeled_proxy_means, unlabeled_proxy_means = _preprocess(y_true, y_proxy, clusters)

    np.testing.assert_array_equal(labeled_true_means, np.array([4.0, 6.0]))
    np.testing.assert_array_equal(labeled_proxy_means, np.array([2.0, 6.0]))
    np.testing.assert_array_equal(unlabeled_proxy_means, np.array([4.0, 6.0]))
