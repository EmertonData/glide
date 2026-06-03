from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def _shuffle(
    array: NDArray,
    rng: np.random.Generator,
) -> Tuple[NDArray, NDArray]:
    order = rng.permutation(len(array))
    shuffled = array[order]
    return shuffled, order


def _compute_cutoff_indices(cumulative_costs: NDArray, order: NDArray, budget: float) -> NDArray:
    cutoff = np.searchsorted(cumulative_costs, budget, side="right")
    kept_indices = order[:cutoff]
    return kept_indices


def _build_output(kept_indices: NDArray, pi_shuffled: NDArray, xi_shuffled: NDArray) -> Tuple[NDArray, NDArray]:
    n = len(pi_shuffled)
    pi_out = np.zeros(n)
    xi_out = np.full(n, np.nan)
    n_kept = len(kept_indices)
    pi_out[kept_indices] = pi_shuffled[:n_kept]
    xi_out[kept_indices] = xi_shuffled[:n_kept]
    return pi_out, xi_out
