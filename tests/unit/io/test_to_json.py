import json

import pytest

from glide.confidence_intervals import CLTConfidenceInterval
from glide.core.mean_inference_result import (
    ClassicalMeanInferenceResult,
    PredictionPoweredMeanInferenceResult,
)
from glide.io import to_json

CI = CLTConfidenceInterval(mean=0.7, std=0.05, confidence_level=0.95)
RESULT = PredictionPoweredMeanInferenceResult(
    confidence_interval=CI,
    metric_name="accuracy",
    estimator_name="Test",
    n_true=10,
    n_proxy=90,
    effective_sample_size=200,
)


def test_to_json_prediction_powered_mean_inference_result():
    parsed = json.loads(to_json(RESULT))
    expected = {
        "metric_name": "accuracy",
        "estimator_name": "Test",
        "n_true": 10,
        "n_proxy": 90,
        "mean": 0.7,
        "std": 0.05,
        "effective_sample_size": 200,
        "confidence_interval": {
            "confidence_level": pytest.approx(0.95, abs=1e-2),
            "lower_bound": pytest.approx(0.6020018007729973, abs=1e-2),
            "upper_bound": pytest.approx(0.7979981992270027, abs=1e-2),
            "width": pytest.approx(0.1959981992270027, abs=1e-2),
        },
    }
    assert parsed == expected


def test_to_json_classical():
    result = ClassicalMeanInferenceResult(
        confidence_interval=CI,
        metric_name="accuracy",
        estimator_name="Test",
        n=100,
    )
    parsed = json.loads(to_json(result))
    expected = {
        "metric_name": "accuracy",
        "estimator_name": "Test",
        "n": 100,
        "mean": 0.7,
        "std": 0.05,
        "confidence_interval": {
            "confidence_level": pytest.approx(0.95, abs=1e-2),
            "lower_bound": pytest.approx(0.6020018007729973, abs=1e-2),
            "upper_bound": pytest.approx(0.7979981992270027, abs=1e-2),
            "width": pytest.approx(0.1959981992270027, abs=1e-2),
        },
    }
    assert parsed == expected
