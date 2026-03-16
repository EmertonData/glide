import json

import pytest

from glide.core.clt_confidence_interval import CLTConfidenceInterval
from glide.core.inference_result import InferenceResult
from glide.io import to_json

CI = CLTConfidenceInterval(mean=0.7, std=0.05, confidence_level=0.95)
RESULT = InferenceResult(
    result=CI, metric_name="accuracy", estimator_name="Test", n_true=10, n_proxy=90, effective_sample_size=200
)


def test_to_json():
    parsed = json.loads(to_json(RESULT))
    assert parsed["metric_name"] == "accuracy"
    assert parsed["estimator_name"] == "Test"
    assert parsed["n_true"] == 10
    assert parsed["n_proxy"] == 90
    assert parsed["effective_sample_size"] == 200
    assert parsed["result"]["mean"] == pytest.approx(0.70, abs=1e-2)
    assert parsed["result"]["std"] == pytest.approx(0.05, abs=1e-2)
    assert parsed["result"]["confidence_level"] == pytest.approx(0.95, abs=1e-2)
    assert parsed["result"]["lower_bound"] == pytest.approx(0.60, abs=1e-2)
    assert parsed["result"]["upper_bound"] == pytest.approx(0.80, abs=1e-2)
