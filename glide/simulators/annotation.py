import numpy as np
from numpy.typing import NDArray


def simulate_annotation(
    y_true_oracle: NDArray,
    xi: NDArray,
) -> NDArray:
    """Reveal oracle labels where annotated and mask the rest as NaN.

    Given a full oracle label array and an annotation indicator, returns an array where labels
    are kept for annotated records (``xi == 1``) and set to ``np.nan`` for unannotated ones
    (``xi == 0`` or ``xi == np.nan``). The input arrays are not mutated.

    Parameters
    ----------
    y_true_oracle : NDArray
        Full oracle ground-truth labels for all records.
    xi : NDArray
        Annotation indicator of the same length. A value of ``1`` means the record was sent
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
    if len(y_true_oracle) != len(xi):
        raise ValueError(f"y_true_oracle and xi must have the same length, got {len(y_true_oracle)} and {len(xi)}")
    if np.any(np.isnan(y_true_oracle)):
        raise ValueError("y_true_oracle contains NaN values")
    xi_float = xi.astype(float)
    if not (np.isnan(xi_float) | np.isin(xi_float, [0.0, 1.0])).all():
        raise ValueError("xi must only contain 0, 1, and np.nan values")

    y_true = y_true_oracle.astype(float)
    y_true[xi_float != 1] = np.nan
    return y_true
