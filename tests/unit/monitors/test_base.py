from unittest.mock import patch

import numpy as np
import pytest

import glide.monitors.base as base_module
from glide.monitors import AsymptoticPPRM


@pytest.fixture
def y_true():
    return np.array([0.49, 0.51, np.nan, np.nan, 0.6, 0.64, np.nan, np.nan])


@pytest.fixture
def y_proxy():
    return np.array([0.5, 0.5, 0.49, 0.55, 0.6, 0.58, 0.6, 0.62])


@pytest.fixture
def batches():
    return np.array([0, 0, 0, 0, 1, 1, 1, 1])


@pytest.fixture
def monitor():
    monitor = AsymptoticPPRM()
    return monitor


# --- _preprocess_subset ---


def test_preprocess_subset_delegates(monitor, y_true, y_proxy, batches):
    mask = ~batches.astype(bool)
    with patch.object(monitor._engine, "preprocess", wraps=monitor._engine.preprocess) as mock_preprocess:
        monitor._preprocess_subset([y_true, y_proxy], mask)

        mock_preprocess.assert_called_once()
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0], y_true[mask])
        np.testing.assert_array_equal(mock_preprocess.call_args[0][1], y_proxy[mask])


# --- _detect ---


def test_detect_delegates(monitor, y_true, y_proxy, batches):
    with (
        patch.object(base_module, "_preprocess", wraps=base_module._preprocess) as mock_preprocess,
        patch.object(monitor, "_preprocess_subset", wraps=monitor._preprocess_subset) as mock_preprocess_subset,
        patch.object(monitor._engine, "fit_tuning_parameter") as mock_fit_tuning_parameter,
        patch.object(monitor._engine, "compute_mean_and_std") as mock_compute_mean_and_std,
        patch.object(
            base_module, "_compute_asymptotic_bounds", wraps=base_module._compute_asymptotic_bounds
        ) as mock_compute_asymptotic_bounds,
        patch.object(base_module, "_postprocess", wraps=base_module._postprocess) as mock_postprocess,
    ):
        mock_fit_tuning_parameter.side_effect = [0.1, 0.2]
        mock_compute_mean_and_std.return_value = (0.0, 1.0)

        monitor._detect(
            fields=[y_true, y_proxy],
            field_names=["y_true", "y_proxy"],
            batches=batches,
            higher_is_better=False,
            confidence_level=0.8,
            tightest_at_batch=10,
            power_tuning=True,
        )

        mock_preprocess.assert_called_once()
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0][0], y_true)
        np.testing.assert_array_equal(mock_preprocess.call_args[0][0][1], y_proxy)
        assert mock_preprocess.call_args[0][1] == ["y_true", "y_proxy"]
        np.testing.assert_array_equal(mock_preprocess.call_args[0][2], batches)
        assert mock_preprocess.call_args[0][3] is False
        assert mock_preprocess.call_args[0][4] == 0.8

        assert mock_preprocess_subset.call_count == 3
        first_subset_call, second_subset_call, third_subset_call = mock_preprocess_subset.call_args_list
        np.testing.assert_array_equal(first_subset_call.args[1], batches == 0)
        np.testing.assert_array_equal(second_subset_call.args[1], batches == 1)
        np.testing.assert_array_equal(third_subset_call.args[1], batches < 1)

        assert mock_fit_tuning_parameter.call_count == 2
        first_fit_call, second_fit_call = mock_fit_tuning_parameter.call_args_list
        assert first_fit_call.kwargs == {"power_tuning": False}
        np.testing.assert_array_equal(first_fit_call.args[0][0], np.array([0.49, 0.51]))
        assert second_fit_call.kwargs == {"power_tuning": True}
        np.testing.assert_array_equal(second_fit_call.args[0][0], np.array([0.49, 0.51]))

        assert mock_compute_mean_and_std.call_count == 2
        first_compute_call, second_compute_call = mock_compute_mean_and_std.call_args_list
        np.testing.assert_array_equal(first_compute_call.args[0][0], np.array([0.49, 0.51]))
        assert first_compute_call.args[1] == 0.1
        np.testing.assert_array_equal(second_compute_call.args[0][0], np.array([0.6, 0.64]))
        assert second_compute_call.args[1] == 0.2

        mock_compute_asymptotic_bounds.assert_called_once()
        np.testing.assert_array_equal(mock_compute_asymptotic_bounds.call_args[0][0], np.array([0.0, 0.0]))
        np.testing.assert_array_equal(mock_compute_asymptotic_bounds.call_args[0][1], np.array([1.0, 1.0]))
        assert mock_compute_asymptotic_bounds.call_args[0][2] == pytest.approx(0.2)
        assert mock_compute_asymptotic_bounds.call_args[0][3] == 10

        mock_postprocess.assert_called_once()
        np.testing.assert_array_equal(mock_postprocess.call_args[0][2], np.array([0.0, 0.0]))
        assert mock_postprocess.call_args[0][3] is False


def test_detect_raises_with_batch_identity_on_too_few_samples(monitor, y_true, y_proxy):
    batches = np.array([0, 0, 0, 0, 0, 1, 1, 1])

    with pytest.raises(ValueError, match=r"Too few labeled or unlabeled samples in dataset\. \(batch '1'\)"):
        monitor._detect(
            fields=[y_true, y_proxy],
            field_names=["y_true", "y_proxy"],
            batches=batches,
            higher_is_better=False,
            confidence_level=0.8,
            tightest_at_batch=10,
            power_tuning=True,
        )
