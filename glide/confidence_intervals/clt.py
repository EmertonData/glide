from dataclasses import dataclass, field
from typing import Literal, Tuple

from scipy.stats import norm


@dataclass
class CLTConfidenceInterval:
    """Confidence interval based on the Central Limit Theorem.

    Constructs a symmetric interval around the point estimate using the standard
    normal distribution: [mean - z * std, mean + z * std], where z is the critical
    value from the standard normal distribution corresponding to the target
    confidence level.

    Parameters
    ----------
    mean : float
        The point estimate of the population mean.
    std : float
        The standard error (standard deviation of the estimate).
    confidence_level : float, optional
        Target coverage probability, e.g. 0.95 for a 95% CI. Default is 0.95.

    Examples
    --------
    >>> from glide.confidence_intervals import CLTConfidenceInterval
    >>> ci = CLTConfidenceInterval(mean=5.0, std=0.2, confidence_level=0.95)
    >>> print(f"[{ci.lower_bound:.3f}, {ci.upper_bound:.3f}]")
    [4.608, 5.392]
    """

    mean: float
    std: float
    confidence_level: float = 0.95
    var: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.var = self.std**2

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

    def test_null_hypothesis(
        self, h0_value: float, alternative: Literal["larger", "smaller", "two-sided"] = "two-sided"
    ) -> Tuple[float, float, float]:
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
            ``(z_stat, p_value, df)`` where ``z_stat`` is the test statistic
            (mean - h0_value) / std, ``p_value`` is the p-value under the standard
            normal distribution, and ``df`` is ``float('inf')``.

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
