from dataclasses import dataclass
from typing import Tuple

from scipy.stats import norm


@dataclass
class CLTConfidenceInterval:
    mean: float
    std: float
    confidence_level: float = 0.95

    def _z_score(self) -> float:
        z_score = norm.ppf((1 + self.confidence_level) / 2)
        return z_score

    @property
    def lower_bound(self) -> float:
        result = self.mean - self.std * self._z_score()
        return result

    @property
    def upper_bound(self) -> float:
        result = self.mean + self.std * self._z_score()
        return result

    def test_null_hypothesis(self, h0_value: float, alternative: str = "two-sided") -> Tuple[float, float, float]:
        """Perform a one-sample z-test against a null hypothesis value.

        Parameters
        ----------
        h0_value : float
            The hypothesized population mean under the null hypothesis (H0: μ = h0_value).
        alternative : str, optional
            The alternative hypothesis. One of:
            - ``'two-sided'`` (default): H1: μ ≠ h0_value
            - ``'larger'``: H1: μ > h0_value
            - ``'smaller'``: H1: μ < h0_value

        Returns
        -------
        Tuple[float, float, float]
            A tuple of (z_stat, p_value, df) where:
            - ``z_stat``: the test statistic (mean - h0_value) / std
            - ``p_value``: the p-value under the standard normal distribution
            - ``df``: degrees of freedom, always ``float('inf')`` for a z-test

        Raises
        ------
        ValueError
            If ``alternative`` is not one of ``'two-sided'``, ``'larger'``, or ``'smaller'``.
        """
        z_stat = (self.mean - h0_value) / self.std
        if alternative == "two-sided":
            p_value = 2 * norm.sf(abs(z_stat))
        elif alternative == "larger":
            p_value = norm.sf(z_stat)
        elif alternative == "smaller":
            p_value = norm.cdf(z_stat)
        else:
            raise ValueError(f"alternative must be 'two-sided', 'larger', or 'smaller', got '{alternative}'")
        df = float("inf")
        return z_stat, p_value, df
