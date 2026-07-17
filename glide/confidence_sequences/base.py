from dataclasses import dataclass
from typing import Literal

from numpy.typing import NDArray

from glide.core.validation import _validate_literal


@dataclass
class ConfidenceSequence:
    """Anytime-valid confidence sequence on a running mean of per-batch estimates.

    Holds the per-look running means and the one-sided anytime-valid bound on the
    side where drift is harmful. The bounds hold simultaneously at all looks, so
    testing after every batch does not inflate the false-alarm probability.

    Parameters
    ----------
    running_mean_estimates : NDArray
        Per-look running mean of the per-batch estimates, in original metric units.
    confidence_bounds : NDArray
        Per-look harmful-side anytime-valid bound, in original metric units.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.confidence_sequences import ConfidenceSequence
    >>> sequence = ConfidenceSequence(
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
