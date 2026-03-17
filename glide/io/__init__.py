import json
from dataclasses import asdict

from glide.core.inference_result import InferenceResult


def to_json(result: InferenceResult) -> str:
    data = asdict(result)
    data["result"]["lower_bound"] = result.result.lower_bound
    data["result"]["upper_bound"] = result.result.upper_bound
    json_str = json.dumps(data, indent=2)
    return json_str
