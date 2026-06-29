from dataclasses import dataclass

from glide.mean_inference_results import MeanInferenceResult


@dataclass(repr=False)
class ClassicalMeanInferenceResult(MeanInferenceResult):
    """Mean inference result for classical (non-bootstrap) methods."""

    n: int = 0

    def __str__(self) -> str:
        lines = self._common_lines() + [
            f"n: {self.n}",
        ]
        return "\n".join(lines)
