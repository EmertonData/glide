from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import brentq

from glide.core.validation import _validate_literal


def _compute_mixture_wealth(deviation: float, variance_process: float) -> float:
    # ψ_E(β) = −log(1−β) − β is the cumulant of the exponential-family mixture.
    def integrand(betting_parameter: float) -> float:
        psi_exponential = -np.log1p(-betting_parameter) - betting_parameter
        value = np.exp(betting_parameter * deviation - psi_exponential * variance_process)
        return value

    integral, _ = quad(integrand, 0.0, 1.0)
    return integral


def _compute_mixture_boundary(variance_process: float, miscoverage: float) -> float:
    wealth_target = 1.0 / miscoverage

    def excess_wealth(deviation: float) -> float:
        value = _compute_mixture_wealth(deviation, variance_process) - wealth_target
        return value

    # No fixed upper bracket works because the root grows with variance_process and
    # 1/miscoverage; double until excess_wealth turns non-negative.
    upper_bracket = 1.0
    while excess_wealth(upper_bracket) < 0.0:
        upper_bracket *= 2.0
    boundary = brentq(excess_wealth, 0.0, upper_bracket)
    return boundary


def _compute_empirical_bernstein_bounds(
    batch_estimates: NDArray,
    seed_center: float,
    miscoverage: float,
) -> Tuple[NDArray, NDArray]:
    n_batches = len(batch_estimates)
    batch_counts = np.arange(1, n_batches + 1)
    running_mean_estimates = np.cumsum(batch_estimates) / batch_counts
    # Centers must be predictable (known before each batch arrives): use the previous
    # running mean, seeded with seed_center before the first batch.
    predictable_centers = np.hstack([np.array([seed_center]), running_mean_estimates[:-1]])
    variance_process = np.cumsum((batch_estimates - predictable_centers) ** 2)
    boundaries = np.array([_compute_mixture_boundary(value, miscoverage) for value in variance_process])
    lower_bounds = running_mean_estimates - boundaries / batch_counts
    return running_mean_estimates, lower_bounds


@dataclass
class EmpiricalBernsteinConfidenceSequence:
    """Anytime-valid empirical-Bernstein confidence sequence on a running mean.

    Holds the per-look running means and the one-sided anytime-valid bound on the
    side where drift is harmful (a lower bound for a risk, an upper bound for a
    performance, after the monitor has mapped the sequence back to the original
    metric orientation). The bounds hold simultaneously at all looks, so testing
    after every batch does not inflate the false-alarm probability.

    Parameters
    ----------
    running_mean_estimates : NDArray
        Per-look running mean of the per-batch estimates, in original metric units.
    confidence_bounds : NDArray
        Per-look harmful-side anytime-valid bound, in original metric units.

    References
    ----------
    Waudby-Smith, Ian, and Aaditya Ramdas. "Estimating means of bounded random
    variables by betting." Journal of the Royal Statistical Society Series B:
    Statistical Methodology 86, no. 1 (2024): 1-27.

    Howard, Steven R., Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. "Time-uniform,
    nonparametric, nonasymptotic confidence sequences." The Annals of Statistics 49,
    no. 2 (2021): 1055-1080.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
    >>> sequence = EmpiricalBernsteinConfidenceSequence(
    ...     running_mean_estimates=np.array([0.4, 0.6]),
    ...     confidence_bounds=np.array([0.1, 0.55]),
    ... )
    >>> sequence.test_null_hypothesis(0.5, alternative="larger")
    array([False,  True])
    """

    running_mean_estimates: NDArray
    confidence_bounds: NDArray

    def test_null_hypothesis(
        self,
        h0_value: float,
        alternative: Literal["larger", "smaller"] = "larger",
    ) -> NDArray:
        """Test the running mean against ``h0_value`` at every look.

        Parameters
        ----------
        h0_value : float
            The threshold the harmful-side bound is tested against (for a monitor,
            the user-supplied business threshold).
        alternative : str, optional
            ``'larger'`` (default) when the metric is a risk: alarm where the lower
            bound exceeds ``h0_value``. ``'smaller'`` when it is a performance: alarm
            where the upper bound falls below ``h0_value``. A confidence sequence is
            one-sided, so ``'two-sided'`` is not accepted.

        Returns
        -------
        NDArray
            Boolean per-look alarm vector, ``True`` once the bound has crossed
            ``h0_value``. Time-uniform family-wise error is controlled over all looks.

        Raises
        ------
        ValueError
            If ``alternative`` is not ``'larger'`` or ``'smaller'``.
        """
        alternatives = ["larger", "smaller"]
        _validate_literal(alternative, "alternative", alternatives)
        if alternative == alternatives[0]:
            alarms = self.confidence_bounds > h0_value
        else:
            alarms = self.confidence_bounds < h0_value
        return alarms
