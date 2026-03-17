import json

import pytest

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.inference_result import SemiSupervisedMeanInferenceResult
from glide.io import to_json

CI = CLTConfidenceInterval(mean=0.7, std=0.05, confidence_level=0.95)
RESULT = SemiSupervisedMeanInferenceResult(
    confidence_interval=CI,
    metric_name="accuracy",
    estimator_name="Test",
    n_true=10,
    n_proxy=90,
    effective_sample_size=200,
)


def test_to_json():
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
            "mean": pytest.approx(0.70, abs=1e-2),
            "std": pytest.approx(0.05, abs=1e-2),
            "confidence_level": pytest.approx(0.95, abs=1e-2),
            "lower_bound": pytest.approx(0.6020018007729973, abs=1e-2),
            "upper_bound": pytest.approx(0.7979981992270027, abs=1e-2),
        },
    }
    assert parsed == expected
