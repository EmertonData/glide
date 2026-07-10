from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import gammainc, gammaln

from glide.core.validation import _validate_bounds, _validate_literal


def _compute_mixture_wealth(deviation: float, variance_process_value: float) -> float:
    exponent_argument = deviation + variance_process_value
    if exponent_argument == 0.0:
        return 1.0
    log_wealth = (
        exponent_argument
        - (variance_process_value + 1) * np.log(exponent_argument)
        + gammaln(variance_process_value + 1)
        + np.log(gammainc(variance_process_value + 1, exponent_argument))
    )
    wealth = np.exp(log_wealth)
    return wealth


def _compute_mixture_boundary(variance_process_value: float, miscoverage: float, upper_bracket: float) -> float:
    wealth_target = 1.0 / miscoverage

    def excess_wealth(deviation: float) -> float:
        value = _compute_mixture_wealth(deviation, variance_process_value) - wealth_target
        return value

    if excess_wealth(upper_bracket) < 0:
        return upper_bracket

    boundary = brentq(excess_wealth, 0.0, upper_bracket)
    return boundary


def _compute_empirical_bernstein_bounds(
    batch_estimates: NDArray,
    seed_center: float,
    miscoverage: float,
) -> Tuple[NDArray, NDArray]:
    _validate_bounds(miscoverage, "miscoverage", lower=0.0, upper=1.0, left_inclusive=False, right_inclusive=False)
    n_batches = len(batch_estimates)
    batch_counts = np.arange(1, n_batches + 1)
    running_mean_estimates = np.cumsum(batch_estimates) / batch_counts
    predictable_centers = np.hstack([np.array([seed_center]), running_mean_estimates[:-1]])
    variance_process = np.cumsum((batch_estimates - predictable_centers) ** 2)
    boundaries = np.array(
        [
            _compute_mixture_boundary(
                variance_process[i], miscoverage, upper_bracket=running_mean_estimates[i] * batch_counts[i]
            )
            for i in range(n_batches)
        ]
    )
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
