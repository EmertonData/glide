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
    """
    data = asdict(result)
    data["mean"] = result.mean
    data["std"] = result.std
    data["confidence_interval"]["lower_bound"] = result.confidence_interval.lower_bound
    data["confidence_interval"]["upper_bound"] = result.confidence_interval.upper_bound
    json_str = json.dumps(data, indent=2)
    return json_str
