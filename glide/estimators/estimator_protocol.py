from typing import Protocol

from glide.core.dataset import Dataset
from glide.core.inference_result import InferenceResult


class Estimator(Protocol):
    def estimate(self, dataset: Dataset) -> InferenceResult: ...
