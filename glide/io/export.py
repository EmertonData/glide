import json
from dataclasses import asdict

from glide.core.mean_inference_result import MeanInferenceResult


def to_json(result: MeanInferenceResult) -> str:
    """Convert a MeanInferenceResult to a JSON string representation.

    Parameters
    ----------
    result : MeanInferenceResult
        The inference result object containing mean, standard deviation, and confidence interval.

    Returns
    -------
    str
        A JSON-formatted string representation of the inference result with 2-space indentation.

    Examples
    --------
    >>> from glide.io import to_json
    >>> from glide.core.mean_inference_result import MeanInferenceResult
    >>> from glide.core.clt_confidence_interval import CLTConfidenceInterval
    >>> confidence_interval = CLTConfidenceInterval(mean=0, std=1)
    >>> inference_result = MeanInferenceResult(confidence_interval=confidence_interval, \
    metric_name="metric", estimator_name="none")
    >>> print(to_json(inference_result))  # doctest: +ELLIPSIS
    {
      "confidence_interval": {
        "mean": 0,
        "std": 1,
        "confidence_level": 0.95,
        "lower_bound": -1.959963984540054,
        "upper_bound": 1.959963984540054
      },
      "metric_name": "metric",
      "estimator_name": "none",
      "mean": 0,
      "std": 1
    }
    """
    data = asdict(result)
    data["mean"] = result.mean
    data["std"] = result.std
    data["confidence_interval"]["lower_bound"] = result.confidence_interval.lower_bound
    data["confidence_interval"]["upper_bound"] = result.confidence_interval.upper_bound
    json_str = json.dumps(data, indent=2)
    return json_str
