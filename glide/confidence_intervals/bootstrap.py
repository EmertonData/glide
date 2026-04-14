from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class BootstrapConfidenceInterval:
    """Quantile bootstrap confidence interval.

    Stores the full distribution of bootstrap point estimates and derives
    bounds as quantiles of that distribution. Supports non-Gaussian and
    asymmetric confidence intervals.

    Parameters
    ----------
    bootstrap_estimates : NDArray
        Array of shape (B,) containing the B bootstrap point estimates.
    confidence_level : float, optional
        Target coverage probability, default 0.95 for 95% CI.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.confidence_intervals import BootstrapConfidenceInterval
    >>> rng = np.random.default_rng(0)
    >>> estimates = rng.normal(loc=5.0, scale=0.3, size=20)
    >>> ci = BootstrapConfidenceInterval(bootstrap_estimates=estimates, confidence_level=0.95)
    >>> 4.0 < ci.lower_bound < ci.mean < ci.upper_bound < 6.0
    True
    """

    bootstrap_estimates: NDArray
    confidence_level: float = 0.95
    mean: float = field(init=False, repr=False)
    var: float = field(init=False, repr=False)
    std: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mean = float(np.mean(self.bootstrap_estimates))
        self.var = float(np.var(self.bootstrap_estimates, ddof=1))
        self.std = float(np.sqrt(self.var))

    @property
    def lower_bound(self) -> float:
        tail = (1 - self.confidence_level) / 2
        bound = float(np.quantile(self.bootstrap_estimates, tail))
        return bound

    @property
    def upper_bound(self) -> float:
        tail = (1 - self.confidence_level) / 2
        bound = float(np.quantile(self.bootstrap_estimates, 1 - tail))
        return bound

    def test_null_hypothesis(
        self,
        h0_value: float,
        alternative: str = "two-sided",
    ) -> Tuple[float, float, float]:
        """Bootstrap hypothesis test against a null value.

        Computes a p-value as the proportion of bootstrap estimates that are
        at least as extreme as `h0_value` under the specified alternative.

        Parameters
        ----------
        h0_value : float
            Hypothesised population mean under H0.
        alternative : str, optional
            One of ``'two-sided'``, ``'larger'``, or ``'smaller'``.

        Returns
        -------
        Tuple[float, float, float]
            ``(test_statistic, p_value, df)`` where ``test_statistic`` is the
            point estimate (mean of bootstrap distribution), ``p_value`` is the
            bootstrap p-value, and ``df`` is ``float('inf')``.
        """
        if alternative == "two-sided":
            centered = np.abs(self.bootstrap_estimates - self.mean)
            observed_deviation = abs(self.mean - h0_value)
            is_at_least_as_extreme = centered >= observed_deviation
        elif alternative == "larger":
            is_at_least_as_extreme = self.bootstrap_estimates <= h0_value
        elif alternative == "smaller":
            is_at_least_as_extreme = self.bootstrap_estimates >= h0_value
        else:
            raise ValueError(f"alternative must be 'two-sided', 'larger', or 'smaller', got '{alternative}'")
        p_value = float(np.mean(is_at_least_as_extreme))
        return self.mean, p_value, float("inf")
