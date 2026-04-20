from typing import Literal, Protocol, Tuple


class ConfidenceInterval(Protocol):
    """Structural protocol for confidence intervals.

    Any class implementing this protocol can be used as a confidence_interval
    in result objects like PredictionPoweredMeanInferenceResult.
    """

    confidence_level: float
    mean: float
    std: float

    @property
    def lower_bound(self) -> float:
        """Lower bound of the confidence interval."""
        ...

    @property
    def upper_bound(self) -> float:
        """Upper bound of the confidence interval."""
        ...

    @property
    def width(self) -> float:
        """Width of the confidence interval."""
        ...

    def test_null_hypothesis(
        self,
        h0_value: float,
        alternative: Literal["larger", "smaller", "two-sided"] = "two-sided",
    ) -> Tuple[float, float, float]:
        """Test null hypothesis against a value.

        Returns (test_statistic, p_value, df).
        """
        ...
