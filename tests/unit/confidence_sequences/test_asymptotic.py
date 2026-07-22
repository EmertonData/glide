from unittest.mock import patch

import numpy as np
import pytest

import glide.confidence_sequences.asymptotic as asymptotic_module
from glide.confidence_sequences.asymptotic import _compute_asymptotic_bounds

# --- _compute_asymptotic_bounds ---


@pytest.fixture
def batch_estimates():
    return np.array([0.4, 0.6])


@pytest.fixture
def batch_std_estimates():
    return np.array([0.1, 0.2])


def test_compute_asymptotic_bounds_delegates_to_validation(batch_estimates, batch_std_estimates):
    with (
        patch.object(asymptotic_module, "_validate_non_empty") as mock_validate_non_empty,
        patch.object(asymptotic_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
        patch.object(asymptotic_module, "_validate_is_integer") as mock_validate_is_integer,
        patch.object(asymptotic_module, "_validate_bounds") as mock_validate_bounds,
    ):
        _compute_asymptotic_bounds(batch_estimates, batch_std_estimates, miscoverage=0.2, tightest_at_batch=1)

        mock_validate_non_empty.assert_called_once_with(batch_estimates, "batch_estimates")
        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], batch_estimates)
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], batch_std_estimates)
        assert mock_validate_equal_lengths.call_args[1] == {"names": ["batch_estimates", "batch_std_estimates"]}
        mock_validate_is_integer.assert_called_once_with(1, "tightest_at_batch")

        assert mock_validate_bounds.call_count == 4
        np.testing.assert_array_equal(mock_validate_bounds.call_args_list[0][0][0], batch_std_estimates)
        assert mock_validate_bounds.call_args_list[0][0][1] == "batch_std_estimates"
        assert mock_validate_bounds.call_args_list[0][1]["lower"] == 0.0
        assert mock_validate_bounds.call_args_list[1][0] == (0.2, "miscoverage")
        assert mock_validate_bounds.call_args_list[1][1] == {
            "lower": 0.0,
            "upper": 0.5,
            "left_inclusive": False,
            "right_inclusive": False,
        }
        assert mock_validate_bounds.call_args_list[2][0] == (1, "tightest_at_batch")
        assert mock_validate_bounds.call_args_list[2][1] == {"lower": 1}
        assert mock_validate_bounds.call_args_list[3][0][1] == "batch_std_estimates"
        assert "must accumulate a positive variance" in mock_validate_bounds.call_args_list[3][1]["error_message"]


def test_compute_asymptotic_bounds(batch_estimates, batch_std_estimates):
    running_means, lower_bounds = _compute_asymptotic_bounds(
        batch_estimates, batch_std_estimates, miscoverage=0.2, tightest_at_batch=1
    )
    np.testing.assert_allclose(running_means, np.array([0.4, 0.5]))
    np.testing.assert_allclose(lower_bounds, np.array([0.181036, 0.247749]), atol=1e-6)


def test_compute_asymptotic_bounds_target_clamped_to_last_batch(batch_estimates, batch_std_estimates):
    _, clamped = _compute_asymptotic_bounds(batch_estimates, batch_std_estimates, miscoverage=0.2, tightest_at_batch=5)
    _, exact = _compute_asymptotic_bounds(batch_estimates, batch_std_estimates, miscoverage=0.2, tightest_at_batch=2)
    np.testing.assert_allclose(clamped, exact)
