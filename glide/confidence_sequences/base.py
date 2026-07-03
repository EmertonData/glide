from typing import Literal, Protocol

from numpy.typing import NDArray


class ConfidenceSequence(Protocol):
    """Structural protocol for anytime-valid confidence sequences.

    The time-uniform analogue of :class:`ConfidenceInterval`: instead of a single
    bound it carries a bound per look (one per batch), valid simultaneously at all
    looks. Any class implementing this protocol can be used as a
    ``confidence_sequence`` in result objects like ``MeanMonitoringResult``.
    """

    running_mean_estimates: NDArray
    confidence_bounds: NDArray

    def test_null_hypothesis(
        self,
        h0_value: float,
        alternative: Literal["larger", "smaller"] = "larger",
    ) -> NDArray:
        """Test the running mean against a value at every look.

        Returns the boolean per-look alarm vector, with time-uniform family-wise
        error control over all looks.
        """
        ...
