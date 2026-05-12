from numpy.typing import NDArray


def simulate_annotation(
    y_true_oracle: NDArray,
    xi: NDArray,
) -> NDArray:
    """Reveal oracle labels where annotated and mask the rest as NaN.

    Given a full oracle label array and a binary annotation indicator, returns
    a float array where labels are kept for annotated records (``xi == 1``) and
    set to ``float("nan")`` for unannotated ones (``xi == 0``). The input arrays
    are not mutated.

    Parameters
    ----------
    y_true_oracle : NDArray
        Full oracle ground-truth labels for all records.
    xi : NDArray
        Binary annotation indicator of the same length. A value of ``1``
        means the record was sent to a human annotator; ``0`` means it was not.

    Returns
    -------
    NDArray
        Float array of the same length as ``y_true_oracle``, with oracle values
        where ``xi == 1`` and ``float("nan")`` where ``xi == 0``.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.simulators import simulate_annotation
    >>> y_true_oracle = np.array([0, 1, 1, 0])
    >>> xi = np.array([1, 0, 1, 0])
    >>> simulate_annotation(y_true_oracle, xi)
    array([ 0., nan,  1., nan])
    """
    y_true = y_true_oracle.astype(float)
    y_true[xi == 0] = float("nan")
    return y_true
