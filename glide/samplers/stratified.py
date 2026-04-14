from typing import Dict, Hashable, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class StratifiedSampler:
    """Sampler for per-stratum annotation budget allocation.

    This class implements stratified sampling strategies that determine how many samples
    to annotate in each stratum, given a fixed annotation budget and proxy labels for
    all samples (provided as numpy arrays). It supports two allocation strategies:

    - **Proportional allocation** (baseline): Allocates budget proportionally to stratum
      sizes, resulting in uniform sampling probabilities across the dataset.

    - **Neyman allocation** (default, optimal): Assigns more budget to strata with higher
      proxy variance, minimising the asymptotic variance of downstream estimators.
      Particularly effective when proxy variance varies substantially across strata.

    Both allocators use largest-remainder rounding (Hamilton's method) to allocate budget
    across strata. Per-stratum sample sizes are capped at stratum size, so total allocated
    budget Σ n_h ≤ budget (may be less if strata are small). The sampler is typically used
    upstream of statistical estimators to plan annotation effort.

    References
    ----------
    Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona.
    "A framework for efficient model evaluation through stratification, sampling,
    and estimation." In European Conference on Computer Vision, pp. 140-158.
    Cham: Springer Nature Switzerland, 2024. https://arxiv.org/abs/2406.07320

    Examples
    --------
    >>> import numpy as np
    >>> from glide.samplers.stratified import StratifiedSampler
    >>> y_proxy = np.array([0.8, 0.9, 0.85, 0.88, 0.2, 0.3])
    >>> groups = np.array(["A", "A", "A", "A", "B", "B"], dtype=object)
    >>> sampler = StratifiedSampler()
    >>> pi, xi = sampler.sample(y_proxy, groups, budget=2, random_seed=1)
    >>> pi
    array([0.25, 0.25, 0.25, 0.25, 0.5 , 0.5 ])
    >>> xi
    array([0, 1, 0, 1, 0, 0])
    """

    def _validate(
        self,
        y_proxy: NDArray,
        groups: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        if np.isnan(y_proxy).any():
            raise ValueError("Input proxy values contain NaN")

        unique_strata = np.unique(groups)
        for stratum_id in unique_strata:
            stratum_mask = groups == stratum_id
            stratum_size = stratum_mask.sum()
            if stratum_size < 2:
                raise ValueError(f"Stratum '{stratum_id}' has fewer than 2 records; std(ddof=1) requires ≥2.")
            stratum_y_proxy = y_proxy[stratum_mask]
            if len(np.unique(stratum_y_proxy)) < 2:
                raise ValueError(f"Stratum '{stratum_id}' has zero variance in proxy values")

        return y_proxy, groups

    def _apply_largest_remainder_rounding(
        self,
        raw_allocation: Dict[Hashable, float],
        budget: int,
    ) -> Dict[Hashable, int]:
        allocation = {}
        remainders = {}

        for stratum_id, raw_value in raw_allocation.items():
            floor_value = int(np.floor(raw_value))
            allocation[stratum_id] = floor_value
            remainder = raw_value - floor_value
            remainders[stratum_id] = remainder

        current_sum = sum(allocation.values())
        remaining_slots = budget - current_sum

        sorted_strata = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
        for stratum_id, _ in sorted_strata[:remaining_slots]:
            allocation[stratum_id] += 1

        return allocation

    def _neyman_allocation(
        self,
        y_proxy: NDArray,
        groups: NDArray,
        budget: int,
    ) -> Dict[Hashable, int]:

        unique_strata = np.unique(groups)

        weights = {}
        stratum_sizes = {}
        for stratum_id in unique_strata:
            stratum_mask = groups == stratum_id
            stratum_size = stratum_mask.sum()
            stratum_y_proxy = y_proxy[stratum_mask]
            stratum_std = np.std(stratum_y_proxy, ddof=1)
            weight = stratum_size * stratum_std
            weights[stratum_id] = weight
            stratum_sizes[stratum_id] = stratum_size

        total_weight = sum(weights.values())

        raw_allocation = {}
        for stratum_id in unique_strata:
            raw_allocation[stratum_id] = budget * weights[stratum_id] / total_weight

        allocation = self._apply_largest_remainder_rounding(raw_allocation, budget)

        for stratum_id in allocation:
            allocation[stratum_id] = min(allocation[stratum_id], stratum_sizes[stratum_id])

        return allocation

    def _proportional_allocation(
        self,
        groups: NDArray,
        budget: int,
    ) -> Dict[Hashable, int]:

        unique_strata = np.unique(groups)
        total_size = len(groups)

        raw_allocation = {}
        stratum_sizes = {}
        for stratum_id in unique_strata:
            stratum_mask = groups == stratum_id
            stratum_size = stratum_mask.sum()
            raw_allocation[stratum_id] = budget * stratum_size / total_size
            stratum_sizes[stratum_id] = stratum_size

        allocation = self._apply_largest_remainder_rounding(raw_allocation, budget)

        for stratum_id in allocation:
            allocation[stratum_id] = min(allocation[stratum_id], stratum_sizes[stratum_id])

        return allocation

    def sample(
        self,
        y_proxy: NDArray,
        groups: NDArray,
        budget: int,
        strategy: Literal["proportional", "neyman"] = "neyman",
        random_seed: Optional[int] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Allocate annotation budget across strata and perform stratified sampling.

        Computes per-stratum sample sizes using the specified allocation strategy and performs
        Bernoulli sampling for each sample based on its stratum's allocation. Neyman allocation
        (default) assigns more budget to strata with higher proxy variance, minimising asymptotic
        variance of downstream estimators. Proportional allocation allocates budget proportionally
        to stratum sizes and serves as a baseline.

        Each sample receives a drawing probability π_i = n_h / stratum_size (capped at 1), and
        is independently selected via a Bernoulli trial. The actual number of selected items is
        a random variable with expectation ≤ budget.

        Parameters
        ----------
        y_proxy : NDArray
            Proxy labels for all samples. Must be 1-dimensional.
        groups : NDArray
            Stratum identifiers for all samples. Must be 1-dimensional with same length as y_proxy.
        budget : int
            Target annotation budget. Must be positive. Mandatory.
        strategy : str, optional
            Allocation strategy: "neyman" (default) or "proportional".
            "neyman": assigns more budget to higher-variance strata.
            "proportional": allocates proportionally to stratum sizes.
        random_seed : int or None, optional
            Random seed for reproducible sampling. Defaults to None (non-deterministic).

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple ``(pi, xi)`` where ``pi`` is an array of drawing probabilities
            in ``(0, 1]`` and ``xi`` is an array of Bernoulli selection indicators
            (1 if selected for annotation, 0 otherwise), both of shape ``(N,)``.

        Raises
        ------
        ValueError
            If strategy is unknown, if budget is not a strictly positive integer, or if budget exceeds
            the number of samples in the input.
        """
        if (not isinstance(budget, (int, np.integer))) or isinstance(budget, bool) or budget <= 0:
            raise ValueError(f"'budget' must be a strictly positive integer; got {budget!r}.")
        if budget > len(y_proxy):
            raise ValueError(
                f"'budget' must not exceed the number of samples; "
                f"got budget={budget} but input has {len(y_proxy)} samples."
            )

        y_proxy, groups = self._validate(y_proxy, groups)

        if strategy == "proportional":
            allocation = self._proportional_allocation(groups, budget)
        elif strategy == "neyman":
            allocation = self._neyman_allocation(y_proxy, groups, budget)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Expected 'proportional' or 'neyman'.")

        # Validate that all strata received non-zero allocation
        for stratum_id, n_h in allocation.items():
            if n_h == 0:
                raise ValueError(
                    f"Stratum '{stratum_id}' has zero allocation. All strata must receive at least "
                    f"one annotation slot. Consider increasing the budget or reducing the number of strata."
                )

        rng = np.random.default_rng(random_seed)

        pi = np.zeros(len(y_proxy))
        xi = np.zeros(len(y_proxy), dtype=int)

        for stratum_id in np.unique(groups):
            stratum_mask = groups == stratum_id
            n_h = allocation[stratum_id]
            stratum_size = stratum_mask.sum()
            pi_value = np.minimum(n_h / stratum_size, 1.0)
            pi[stratum_mask] = pi_value
            xi[stratum_mask] = rng.binomial(n=1, p=np.full(stratum_size, pi_value)).astype(float)

        return pi, xi
