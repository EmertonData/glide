from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def _shuffle(
    arrays: Tuple[NDArray, ...],
    rng: np.random.Generator,
) -> Tuple[Tuple[NDArray, ...], NDArray]:
    order = rng.permutation(len(arrays[0]))
    shuffled = tuple(arr[order] for arr in arrays)
    return shuffled, order


def _compute_cutoff_indices(cumulative_costs: NDArray, order: NDArray, budget: float) -> NDArray:
    cutoff = np.searchsorted(cumulative_costs, budget, side="right")
    kept_indices = order[:cutoff]
    return kept_indices


def _build_output(kept_indices: NDArray, pi: NDArray, xi_shuffled: NDArray) -> Tuple[NDArray, NDArray]:
    n = len(pi)
    pi_out = np.zeros(n)
    xi_out = np.full(n, np.nan)
    pi_out[kept_indices] = pi[kept_indices]
    xi_out[kept_indices] = xi_shuffled[: len(kept_indices)]
    return pi_out, xi_out
