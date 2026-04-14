from typing import Protocol, Tuple


class ConfidenceInterval(Protocol):
    """Structural protocol for confidence interval implementations.

    A confidence interval type must provide bounds, central estimate, spread,
    and support hypothesis testing.
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
        """Perform a hypothesis test against a null value.

        Parameters
        ----------
        h0_value : float
            The hypothesized value under the null hypothesis.
        alternative : str, optional
            One of ``'two-sided'``, ``'larger'``, or ``'smaller'``.

        Returns
        -------
        Tuple[float, float, float]
            ``(test_statistic, p_value, df)`` where ``test_statistic`` is a
            summary of the observed data, ``p_value`` is the probability of
            observing a result at least as extreme as the observed one under
            the null hypothesis, and ``df`` is the degrees of freedom.
        """
        ...
