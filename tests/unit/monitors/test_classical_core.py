from unittest.mock import patch

import numpy as np
import pytest

import glide.monitors.classical_core as classical_core_module
from glide.monitors.classical_core import _compute_batch_estimates, _preprocess


@pytest.fixture
def y():
    return np.array([0.49, 0.51, 0.5, 0.54])


@pytest.fixture
def batches():
    return np.array([0, 0, 1, 1])


# --- _preprocess ---


def test_preprocess_delegates_to_validation(y, batches):
    with (
        patch.object(classical_core_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(classical_core_module, "_validate_non_empty") as mock_validate_non_empty,
        patch.object(classical_core_module, "_validate_has_no_nan") as mock_validate_has_no_nan,
        patch.object(classical_core_module, "_validate_bounds") as mock_validate_bounds,
        patch.object(classical_core_module, "_unique_ordered_batches") as mock_unique_ordered_batches,
    ):
        mock_unique_ordered_batches.return_value = (np.array([0, 1]), np.array([0, 0, 1, 1]))
        _preprocess(
            y,
            batches,
            higher_is_better=False,
            threshold=0.5,
            confidence_level=0.8,
            metric_lower_bound=0.0,
            metric_upper_bound=1.0,
        )

        mock_validate_non_empty.assert_called_once()
        np.testing.assert_array_equal(mock_validate_non_empty.call_args[0][0], y)
        assert mock_validate_non_empty.call_args[0][1] == "y"

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], y)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], batches)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["y", "batches"]}

        mock_validate_has_no_nan.assert_called_once()
        np.testing.assert_array_equal(mock_validate_has_no_nan.call_args[0][0], batches)
        assert mock_validate_has_no_nan.call_args[0][1] == "batches"

        assert mock_validate_bounds.call_count == 5

        assert mock_validate_bounds.call_args_list[0][0] == (0.8, "confidence_level")
        assert mock_validate_bounds.call_args_list[0][1] == {
            "lower": 0,
            "upper": 1,
            "left_inclusive": False,
            "right_inclusive": False,
        }

        assert mock_validate_bounds.call_args_list[1][0] == (0.0, "metric_lower_bound")
        assert mock_validate_bounds.call_args_list[1][1]["upper"] == 1.0
        assert mock_validate_bounds.call_args_list[1][1]["right_inclusive"] is False
        assert (
            "'metric_lower_bound' must be strictly smaller than 'metric_upper_bound'"
            in mock_validate_bounds.call_args_list[1][1]["error_message"]
        )

        assert mock_validate_bounds.call_args_list[2][0] == (0.5, "threshold")
        assert mock_validate_bounds.call_args_list[2][1]["lower"] == 0.0
        assert mock_validate_bounds.call_args_list[2][1]["upper"] == 1.0
        assert "'threshold' must lie between" in mock_validate_bounds.call_args_list[2][1]["error_message"]

        mock_unique_ordered_batches.assert_called_once()
        np.testing.assert_array_equal(mock_unique_ordered_batches.call_args[0][0], batches)

        np.testing.assert_array_equal(mock_validate_bounds.call_args_list[3][0][0], y)
        assert mock_validate_bounds.call_args_list[3][0][1] == "y"
        assert mock_validate_bounds.call_args_list[3][1]["lower"] == 0.0
        assert mock_validate_bounds.call_args_list[3][1]["upper"] == 1.0
        assert "'y' values must lie between" in mock_validate_bounds.call_args_list[3][1]["error_message"]

        assert mock_validate_bounds.call_args_list[4][0][0] == 2
        assert mock_validate_bounds.call_args_list[4][0][1] == "y"
        assert mock_validate_bounds.call_args_list[4][1]["lower"] == 2
        assert (
            "'y' must have at least 2 non-NaN values per batch"
            in mock_validate_bounds.call_args_list[4][1]["error_message"]
        )


def test_preprocess_known_output(y, batches):
    risk_y, risk_threshold, batch_codes, batch_n = _preprocess(
        y,
        batches,
        higher_is_better=False,
        threshold=0.5,
        confidence_level=0.8,
        metric_lower_bound=0.0,
        metric_upper_bound=1.0,
    )

    np.testing.assert_allclose(risk_y, y)
    assert risk_threshold == pytest.approx(0.5)
    np.testing.assert_array_equal(batch_codes, np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(batch_n, np.array([2, 2]))


# --- _compute_batch_estimates ---


def test_compute_batch_estimates(y, batches):
    expected_batch_mean_estimates = np.array([0.5, 0.52])
    expected_batch_std_estimates = np.array([0.01, 0.02])

    batch_mean_estimates, batch_std_estimates = _compute_batch_estimates(y, batches)

    np.testing.assert_allclose(batch_mean_estimates, expected_batch_mean_estimates)
    np.testing.assert_allclose(batch_std_estimates, expected_batch_std_estimates, atol=1e-10)
