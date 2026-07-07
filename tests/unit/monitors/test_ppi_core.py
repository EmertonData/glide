import numpy as np
import pytest

from glide.monitors.ppi_core import (
    _compute_clipped_tuning_parameter,
    _denormalize_from_unit_interval,
    _normalize_to_unit_interval,
)

# --- _compute_clipped_tuning_parameter ---


def test_compute_clipped_tuning_parameter_power_tuning_false():
    y_true_labeled = np.array([0.1, 0.3])
    y_proxy_labeled = np.array([0.1, 0.3])
    y_proxy_unlabeled = np.array([0.2, 0.4])

    tuning_parameter = _compute_clipped_tuning_parameter(
        y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, power_tuning=False, max_tuning_parameter=1.0
    )

    assert tuning_parameter == pytest.approx(1.0)


def test_compute_clipped_tuning_parameter_within_range():
    y_true_labeled = np.array([0.1, 0.3])
    y_proxy_labeled = np.array([0.15, 0.25])
    y_proxy_unlabeled = np.array([0.2, 0.4])

    unclipped = _compute_clipped_tuning_parameter(
        y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, power_tuning=True, max_tuning_parameter=10.0
    )
    clipped = _compute_clipped_tuning_parameter(
        y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, power_tuning=True, max_tuning_parameter=1.0
    )

    assert clipped == pytest.approx(unclipped)


def test_compute_clipped_tuning_parameter_clips_to_upper_bound():
    y_true_labeled = np.array([0.0, 1.0])
    y_proxy_labeled = np.array([0.0, 1.0])
    y_proxy_unlabeled = np.array([0.0, 1.0])

    tuning_parameter = _compute_clipped_tuning_parameter(
        y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, power_tuning=True, max_tuning_parameter=0.5
    )

    assert tuning_parameter == pytest.approx(0.5)


def test_compute_clipped_tuning_parameter_clips_to_lower_bound():
    y_true_labeled = np.array([0.1, 0.3])
    y_proxy_labeled = np.array([0.4, 0.1])
    y_proxy_unlabeled = np.array([0.2, 0.4])

    tuning_parameter = _compute_clipped_tuning_parameter(
        y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, power_tuning=True, max_tuning_parameter=1.0
    )

    assert tuning_parameter == pytest.approx(0.0)


# --- _normalize_to_unit_interval / _denormalize_from_unit_interval ---


def test_normalize_to_unit_interval():
    values = np.array([-1.0, 0.0, 2.0])

    normalized = _normalize_to_unit_interval(values, max_tuning_parameter=1.0)

    np.testing.assert_allclose(normalized, np.array([0.0, 1.0 / 3.0, 1.0]))


def test_denormalize_from_unit_interval():
    values = np.array([0.0, 1.0 / 3.0, 1.0])

    original = _denormalize_from_unit_interval(values, max_tuning_parameter=1.0)

    np.testing.assert_allclose(original, np.array([-1.0, 0.0, 2.0]))


def test_normalize_denormalize_round_trip():
    values = np.array([-0.3, 0.5, 1.2])

    round_tripped = _denormalize_from_unit_interval(
        _normalize_to_unit_interval(values, max_tuning_parameter=0.7), max_tuning_parameter=0.7
    )

    np.testing.assert_allclose(round_tripped, values)
