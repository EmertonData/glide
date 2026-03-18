from typing import Protocol

from glide.core.dataset import Dataset
from glide.core.mean_inference_result import MeanInferenceResult


class MeanEstimator(Protocol):
    def estimate(self, dataset: Dataset) -> MeanInferenceResult: ...
