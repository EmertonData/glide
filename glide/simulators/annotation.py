import numpy as np
from numpy.typing import NDArray

from glide.core.validation import _validate_binary_or_nan, _validate_equal_lengths, _validate_has_no_nan


def simulate_annotation(
    y_true_oracle: NDArray,
    xi: NDArray,
) -> NDArray:
    """Reveal oracle labels where annotated and mask the rest as NaN.

    Given a full oracle label array and an annotation indicator, returns an array where labels
    are kept for annotated elements (``xi == 1``) and set to ``np.nan`` for unannotated ones
    (``xi == 0`` or ``xi == np.nan``). The input arrays are not mutated.

    Parameters
    ----------
    y_true_oracle : NDArray
        Full oracle ground-truth labels for all elements.
    xi : NDArray
        Annotation indicator of the same length. A value of ``1`` means the element was sent
        to a human annotator; ``0`` or ``np.nan`` means it was not.

    Returns
    -------
    NDArray
        Array of the same length as ``y_true_oracle``, with oracle values where ``xi == 1``
        and ``np.nan`` where ``xi == 0`` or ``xi == np.nan``.

    Raises
    ------
    ValueError
        If ``y_true_oracle`` and ``xi`` have different lengths.
    ValueError
        If ``y_true_oracle`` contains NaN values.
    ValueError
        If ``xi`` contains values other than ``0``, ``1``, and ``np.nan``.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import simulate_annotation
    >>> y_true_oracle = np.array([0, 1, 1, 0])
    >>> xi = np.array([1, 0, 1, np.nan])
    >>> simulate_annotation(y_true_oracle, xi)
    array([ 0., nan,  1., nan])
    """
    _validate_equal_lengths(y_true_oracle, xi, names=["y_true_oracle", "xi"])
    _validate_has_no_nan(y_true_oracle, "y_true_oracle")
    xi_float = xi.astype(float)
    _validate_binary_or_nan(xi, "xi")

    y_true = y_true_oracle.astype(float)
    y_true[xi_float != 1] = np.nan
    return y_true
