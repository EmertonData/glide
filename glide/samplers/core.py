from typing import Optional, Tuple, Union

import numpy as np
from numpy.random.bit_generator import SeedSequence
from numpy.typing import NDArray


def _draw_shuffled_bernoulli(
    pi: NDArray,
    random_seed: Optional[Union[int, SeedSequence]] = None,
) -> Tuple[NDArray, NDArray]:
    n_samples = len(pi)
    rng = np.random.default_rng(random_seed)
    order = rng.permutation(n_samples)
    xi_shuffled = rng.binomial(n=1, p=pi[order]).astype(float)
    return order, xi_shuffled


def _apply_budget_cutoff(
    xi_shuffled: NDArray,
    pi: NDArray,
    cumulative_costs: NDArray,
    order: NDArray,
    budget: float,
) -> Tuple[NDArray, NDArray]:
    n_samples = len(order)
    cutoff = np.searchsorted(cumulative_costs, budget, side="right")
    kept_indices = order[:cutoff]

    pi_out = np.zeros(n_samples)
    xi_out = np.full(n_samples, np.nan)

    pi_out[kept_indices] = pi[kept_indices]
    xi_out[kept_indices] = xi_shuffled[:cutoff]
    return pi_out, xi_out
