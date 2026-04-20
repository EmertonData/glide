from dataclasses import dataclass, field
from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class BootstrapConfidenceInterval:
    """Quantile bootstrap confidence interval.

    Stores the full distribution of bootstrap point estimates and derives
    bounds as quantiles of that distribution.

    Parameters
    ----------
    bootstrap_estimates : NDArray
        Array of shape (B,) containing the B bootstrap point estimates.
    confidence_level : float, optional
        Target coverage, e.g. 0.95 for a 95 % CI. Default is 0.95.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.confidence_intervals import BootstrapConfidenceInterval
    >>> rng = np.random.default_rng(0)
    >>> estimates = rng.normal(loc=5.0, scale=0.3, size=20)
    >>> ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates, confidence_level=0.95)
    >>> print(f"[{ci.lower_bound:.3f}, {ci.upper_bound:.3f}]")
    [4.453, 5.354]
    """

    bootstrap_estimates: NDArray
    mean: float = field(init=False, repr=False)
    var: float = field(init=False, repr=False)
    std: float = field(init=False, repr=False)
    _sorted_estimates: NDArray = field(init=False, repr=False)
    _confidence_level: float = field(init=False, repr=False)
    lower_bound: float = field(init=False, repr=False)
    upper_bound: float = field(init=False, repr=False)
    width: float = field(init=False, repr=False)

    def __init__(self, bootstrap_estimates: NDArray, confidence_level: float = 0.95) -> None:
        self.bootstrap_estimates = bootstrap_estimates
        self.mean = float(np.mean(bootstrap_estimates))
        self.var = float(np.var(bootstrap_estimates, ddof=1))
        self.std = float(np.sqrt(self.var))
        self._sorted_estimates = np.sort(bootstrap_estimates)
        self.confidence_level = confidence_level

    @property
    def confidence_level(self) -> float:
        return self._confidence_level

    @confidence_level.setter
    def confidence_level(self, value: float) -> None:
        if not 0 < value < 1:
            raise ValueError(f"confidence_level must be in (0, 1), got {value}")
        self._confidence_level = value
        alpha_over_two = (1 - value) / 2
        self.lower_bound = float(np.quantile(self._sorted_estimates, alpha_over_two))
        self.upper_bound = float(np.quantile(self._sorted_estimates, 1 - alpha_over_two))
        self.width = self.upper_bound - self.lower_bound

    def test_null_hypothesis(
        self,
        h0_value: float,
        alternative: Literal["larger", "smaller", "two-sided"] = "two-sided",
    ) -> Tuple[float, float, float]:
        """Bootstrap hypothesis test against a null value.

        Computes a p-value as the proportion of bootstrap estimates that are
        at least as extreme as `h0_value` under the specified alternative.

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
            ``(test_statistic, p_value, df)`` where ``test_statistic`` is the
            point estimate (mean of bootstrap distribution), ``p_value`` is the
            bootstrap p-value, and ``df`` is ``float('inf')``.
        """
        n = len(self._sorted_estimates)

        if alternative == "two-sided":
            observed_deviation = abs(h0_value - self.mean)
            # Count estimates <= (mean - deviation) or >= (mean + deviation)
            lower_threshold = self.mean - observed_deviation
            upper_threshold = self.mean + observed_deviation
            count_below = np.searchsorted(self._sorted_estimates, lower_threshold, side="right")
            count_above = n - np.searchsorted(self._sorted_estimates, upper_threshold, side="left")
            count_extreme = count_below + count_above
        elif alternative == "larger":
            # Count estimates <= h0_value (evidence against "larger" alternative)
            count_extreme = np.searchsorted(self._sorted_estimates, h0_value, side="right")
        elif alternative == "smaller":
            # Count estimates >= h0_value (evidence against "smaller" alternative)
            count_extreme = n - np.searchsorted(self._sorted_estimates, h0_value, side="left")
        else:
            raise ValueError(f"alternative must be 'two-sided', 'larger', or 'smaller', got '{alternative}'")

        p_value = float(count_extreme) / n

        return self.mean, p_value, float("inf")
