import numpy as np
import pytest

from glide.confidence_sequences import AsymptoticConfidenceSequence
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
    return AsymptoticPPRM()


# --- _preprocess_subset ---


def test_preprocess_subset(monitor, y_true, y_proxy, batches):
    y_true_labeled, y_proxy_labeled, y_proxy_unlabeled = monitor._preprocess_subset(
        [y_true, y_proxy], ~batches.astype(bool)
    )

    np.testing.assert_array_equal(y_true_labeled, np.array([0.49, 0.51]))
    np.testing.assert_array_equal(y_proxy_labeled, np.array([0.5, 0.5]))
    np.testing.assert_array_equal(y_proxy_unlabeled, np.array([0.49, 0.55]))


# --- _detect ---


def test_detect_known_output(monitor, y_true, y_proxy, batches):
    expected_batch_mean_estimates = np.array([0.52, 0.62])
    expected_running_mean_estimates = np.array([0.52, 0.57])
    expected_confidence_bounds = np.array([0.449, 0.529])

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
    np.testing.assert_allclose(confidence_sequence.running_mean_estimates, expected_running_mean_estimates)
    np.testing.assert_allclose(confidence_sequence.confidence_bounds, expected_confidence_bounds, atol=5e-4)


def test_detect_known_output_power_tuning_false(monitor, y_true, y_proxy, batches):
    expected_batch_mean_estimates = np.array([0.52, 0.64])
    expected_running_mean_estimates = np.array([0.52, 0.58])
    expected_confidence_bounds = np.array([0.447, 0.531])

    _, batch_mean_estimates, confidence_sequence = monitor._detect(
        fields=[y_true, y_proxy],
        field_names=["y_true", "y_proxy"],
        batches=batches,
        higher_is_better=False,
        confidence_level=0.8,
        tightest_at_batch=10,
        power_tuning=False,
    )

    np.testing.assert_allclose(batch_mean_estimates, expected_batch_mean_estimates)
    np.testing.assert_allclose(confidence_sequence.running_mean_estimates, expected_running_mean_estimates)
    np.testing.assert_allclose(confidence_sequence.confidence_bounds, expected_confidence_bounds, atol=5e-4)


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
