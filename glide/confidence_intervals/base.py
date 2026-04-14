from typing import Protocol, Tuple


class ConfidenceInterval(Protocol):
    """Structural protocol for confidence intervals.

    Any class implementing this protocol can be used as a confidence_interval
    in result objects like SemiSupervisedMeanInferenceResult.
    """

    confidence_level: float
    mean: float
    std: float
    var: float

    @property
    def lower_bound(self) -> float:
        """Lower bound of the confidence interval."""
        ...

    @property
    def upper_bound(self) -> float:
        """Upper bound of the confidence interval."""
        ...

    def test_null_hypothesis(
        self,
        h0_value: float,
        alternative: str = "two-sided",
    ) -> Tuple[float, float, float]:
        """Test null hypothesis against a value.

        Returns (test_statistic, p_value, df).
        """
        ...
