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
    >>> from glide.core.dataset import Dataset
    >>> from glide.estimators import ClassicalMeanEstimator
    >>> from glide.io import to_json
    >>> dataset = Dataset([{"y_true":i} for i in range(10)])
    >>> estimator = ClassicalMeanEstimator()
    >>> estimation_result = estimator.estimate(dataset, y_field="y_true", confidence_level=0.9)
    >>> print(to_json(estimation_result))
    {
      "confidence_interval": {
        "mean": 4.5,
        "std": 0.9574271077563381,
        "confidence_level": 0.9,
        "lower_bound": 2.9251725492653295,
        "upper_bound": 6.07482745073467
      },
      "metric_name": "Metric",
      "estimator_name": "ClassicalMeanEstimator",
      "n": 10,
      "mean": 4.5,
      "std": 0.9574271077563381
    }
    """
    data = asdict(result)
    data["mean"] = result.mean
    data["std"] = result.std
    data["confidence_interval"]["lower_bound"] = result.confidence_interval.lower_bound
    data["confidence_interval"]["upper_bound"] = result.confidence_interval.upper_bound
    json_str = json.dumps(data, indent=2)
    return json_str
