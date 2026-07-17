from unittest.mock import patch

import numpy as np
import pytest

import glide.confidence_sequences.base as base_module
from glide.confidence_sequences import ConfidenceSequence

# --- ConfidenceSequence ---


@pytest.fixture
def sequence():
    return ConfidenceSequence(
        running_mean_estimates=np.array([0.4, 0.6]),
        confidence_bounds=np.array([0.1, 0.55]),
    )


def test_null_hypothesis_delegates_to_validation(sequence):
    with patch.object(base_module, "_validate_literal") as mock_validate_literal:
        sequence.test_null_hypothesis(0.5, alternative="larger")

        mock_validate_literal.assert_called_once_with("larger", "alternative", ["larger", "smaller"])


def test_null_hypothesis_larger(sequence):
    alarms = sequence.test_null_hypothesis(0.5, alternative="larger")
    np.testing.assert_array_equal(alarms, np.array([False, True]))


def test_null_hypothesis_smaller(sequence):
    alarms = sequence.test_null_hypothesis(0.3, alternative="smaller")
    np.testing.assert_array_equal(alarms, np.array([True, False]))
