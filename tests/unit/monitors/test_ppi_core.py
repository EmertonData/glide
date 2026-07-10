from unittest.mock import patch

import numpy as np
import pytest

from glide.monitors.ppi_core import _compute_clipped_tuning_parameter


# --- fixtures ---
@pytest.fixture
def y_true_labeled():
    return np.array([0.1, 0.3])


@pytest.fixture
def y_proxy_labeled():
    return np.array([0.15, 0.25])


@pytest.fixture
def y_proxy_unlabeled():
    return np.array([0.2, 0.4])


# --- _compute_clipped_tuning_parameter ---


def test_compute_clipped_tuning_parameter_delegates(y_true_labeled, y_proxy_labeled, y_proxy_unlabeled):
    with patch("glide.monitors.ppi_core._compute_tuning_parameter") as mock_compute_tuning_parameter:
        mock_compute_tuning_parameter.return_value = 0.5
        _compute_clipped_tuning_parameter(
            y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, power_tuning=False, max_tuning_parameter=1.0
        )

    mock_compute_tuning_parameter.assert_called_once()
    np.testing.assert_array_equal(mock_compute_tuning_parameter.call_args[0][0], y_true_labeled)
    np.testing.assert_array_equal(mock_compute_tuning_parameter.call_args[0][1], y_proxy_labeled)
    np.testing.assert_array_equal(mock_compute_tuning_parameter.call_args[0][2], y_proxy_unlabeled)
    assert mock_compute_tuning_parameter.call_args[0][3] is False


def test_compute_clipped_tuning_parameter_known_value(y_true_labeled, y_proxy_labeled, y_proxy_unlabeled):
    expected = 0.428
    result = _compute_clipped_tuning_parameter(
        y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, power_tuning=True, max_tuning_parameter=1.0
    )

    assert result == pytest.approx(expected, abs=0.001)
