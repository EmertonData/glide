from typing import Protocol

from glide.core.dataset import Dataset
from glide.core.inference_result import MeanInferenceResult


class Estimator(Protocol):
    def estimate(self, dataset: Dataset) -> MeanInferenceResult: ...
