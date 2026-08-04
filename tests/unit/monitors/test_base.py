from unittest.mock import patch

import numpy as np
import pytest

import glide.monitors.base as base_module
from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.monitors import AsymptoticPPRM


@pytest.fixture
def y_true():
    return np.array([0.49, 0.51, np.nan, np.nan, 0.5, 0.54, np.nan, np.nan])


@pytest.fixture
def y_proxy():
    return np.array([0.5, 0.5, 0.49, 0.55, 0.52, 0.48, 0.5, 0.52])


@pytest.fixture
def batches():
    return np.array([0, 0, 0, 0, 1, 1, 1, 1])


@pytest.fixture
def monitor():
    return AsymptoticPPRM()


# --- _preprocess_subset ---


def test_preprocess_subset(monitor, y_true, y_proxy):
    mask = np.array([True, True, True, True, False, False, False, False])

    y_true_labeled, y_proxy_labeled, y_proxy_unlabeled = monitor._preprocess_subset([y_true, y_proxy], mask)

    np.testing.assert_array_equal(y_true_labeled, np.array([0.49, 0.51]))
    np.testing.assert_array_equal(y_proxy_labeled, np.array([0.5, 0.5]))
    np.testing.assert_array_equal(y_proxy_unlabeled, np.array([0.49, 0.55]))


# --- _detect ---


def test_detect_delegates_to_validation(monitor, y_true, y_proxy, batches):
    with patch.object(base_module, "_validate_bounds") as mock_validate_bounds:
        monitor._detect(
            fields=[y_true, y_proxy],
            field_names=["y_true", "y_proxy"],
            batches=batches,
            higher_is_better=False,
            confidence_level=0.8,
            tightest_at_batch=10,
            power_tuning=True,
        )

    mock_validate_bounds.assert_called_once_with(
        0.8,
        "confidence_level",
        lower=0.5,
        upper=1,
        left_inclusive=False,
        right_inclusive=False,
        error_message="'confidence_level' must be in (0.5, 1) for the asymptotic monitor; got 0.8.",
    )


def test_detect_known_batch_mean_estimates(monitor, y_true, y_proxy, batches):
    expected_batch_mean_estimates = np.array([0.52, 0.52])

    batch_codes, batch_mean_estimates, confidence_sequence = monitor._detect(
        fields=[y_true, y_proxy],
        field_names=["y_true", "y_proxy"],
        batches=batches,
        higher_is_better=False,
        confidence_level=0.8,
        tightest_at_batch=10,
        power_tuning=True,
    )

    np.testing.assert_array_equal(batch_codes, batches)
    np.testing.assert_allclose(batch_mean_estimates, expected_batch_mean_estimates)
    assert isinstance(confidence_sequence, AsymptoticConfidenceSequence)


def test_detect_power_tuning_false(monitor, y_true, y_proxy, batches):
    expected_batch_mean_estimates = np.array([0.52, 0.53])

    _, batch_mean_estimates, _ = monitor._detect(
        fields=[y_true, y_proxy],
        field_names=["y_true", "y_proxy"],
        batches=batches,
        higher_is_better=False,
        confidence_level=0.8,
        tightest_at_batch=10,
        power_tuning=False,
    )

    np.testing.assert_allclose(batch_mean_estimates, expected_batch_mean_estimates)


def test_detect_raises_with_batch_identity_on_too_few_samples(monitor):
    y_true = np.array([0.49, 0.51, np.nan, np.nan, 0.5, 0.6, np.nan])
    y_proxy = np.array([0.5, 0.5, 0.49, 0.55, 0.52, 0.48, 0.5])
    batches = np.array([0, 0, 0, 0, 1, 1, 1])

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
