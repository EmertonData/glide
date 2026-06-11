from typing import Dict, Hashable, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from glide.core.validation import (
    _validate_is_integer,
    _validate_literal,
    _validate_n_samples_bound,
    _validate_strictly_positive,
    _validate_y_proxy,
)


class StratifiedSampler:
    """Sampler for per-stratum annotation allocation.

    This class implements stratified sampling strategies that determine how many samples
    to annotate in each stratum, given a target annotation count (``n_samples``) and proxy labels for
    all samples (provided as numpy arrays). It supports two allocation strategies:

    - **Proportional allocation** (baseline): Allocates budget proportionally to stratum
      sizes, resulting in uniform sampling probabilities across the dataset.

    - **Neyman allocation** (default, optimal): Assigns more budget to strata with higher
      proxy variance, minimising the asymptotic variance of downstream estimators.
      Particularly effective when proxy variance varies substantially across strata.

    Both allocators use largest-remainder rounding (Hamilton's method) to allocate ``n_samples``
    across strata. Per-stratum sample sizes are capped at stratum size, so total allocated count
    Σ n_h ≤ n_samples (may be less if strata are small). The sampler is typically used
    upstream of statistical estimators to plan annotation effort.

    References
    ----------
    Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona. "A framework
    for efficient model evaluation through stratification, sampling, and estimation."
    In European Conference on Computer Vision, pp. 140-158. Cham: Springer Nature
    Switzerland, 2024.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.samplers import StratifiedSampler
    >>> y_proxy = np.array([0.8, 0.9, 0.85, 0.88, 2.4 , 2.5 , 2.45, 2.48])
    >>> groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"], dtype=object)
    >>> sampler = StratifiedSampler()
    >>> xi = sampler.sample(y_proxy, groups, n_samples=4, random_seed=1)
    >>> xi
    array([0, 1, 1, 0, 1, 0, 1, 0])
    """

    def _validate(self, y_proxy: NDArray, groups: NDArray, n_samples: int, strategy: str) -> None:
        _validate_literal(strategy, "strategy", ["proportional", "neyman"])
        _validate_is_integer(n_samples, "n_samples")
        _validate_strictly_positive(n_samples, "n_samples")
        _validate_n_samples_bound(n_samples, len(y_proxy))
        _validate_y_proxy(y_proxy)
        for stratum_id in np.unique(groups):
            stratum_mask = groups == stratum_id
            _validate_y_proxy(y_proxy[stratum_mask], stratum_id)

    def _apply_largest_remainder_rounding(
        self,
        raw_allocation: Dict[Hashable, float],
        n_samples: int,
    ) -> Dict[Hashable, int]:
        allocation = {}
        remainders = {}

        for stratum_id, raw_value in raw_allocation.items():
            floor_value = int(np.floor(raw_value))
            allocation[stratum_id] = floor_value
            remainder = raw_value - floor_value
            remainders[stratum_id] = remainder

        current_sum = sum(allocation.values())
        remaining_slots = n_samples - current_sum

        sorted_strata = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
        for stratum_id, _ in sorted_strata[:remaining_slots]:
            allocation[stratum_id] += 1

        return allocation

    def _neyman_allocation(
        self,
        y_proxy: NDArray,
        groups: NDArray,
        n_samples: int,
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
            raw_allocation[stratum_id] = n_samples * weights[stratum_id] / total_weight

        allocation = self._apply_largest_remainder_rounding(raw_allocation, n_samples)

        for stratum_id in allocation:
            allocation[stratum_id] = min(allocation[stratum_id], stratum_sizes[stratum_id])

        return allocation

    def _proportional_allocation(
        self,
        groups: NDArray,
        n_samples: int,
    ) -> Dict[Hashable, int]:

        unique_strata = np.unique(groups)
        total_size = len(groups)

        raw_allocation = {}
        stratum_sizes = {}
        for stratum_id in unique_strata:
            stratum_mask = groups == stratum_id
            stratum_size = stratum_mask.sum()
            raw_allocation[stratum_id] = n_samples * stratum_size / total_size
            stratum_sizes[stratum_id] = stratum_size

        allocation = self._apply_largest_remainder_rounding(raw_allocation, n_samples)

        for stratum_id in allocation:
            allocation[stratum_id] = min(allocation[stratum_id], stratum_sizes[stratum_id])

        return allocation

    def sample(
        self,
        y_proxy: NDArray,
        groups: NDArray,
        n_samples: int,
        strategy: Literal["proportional", "neyman"] = "neyman",
        random_seed: Optional[int] = None,
    ) -> NDArray:
        """Allocate annotation budget across strata and perform stratified sampling.

        Computes allocated annotation counts ``n_h`` for each stratum ``h`` using the
        specified allocation strategy and selects exactly ``n_h`` samples from each stratum
        without replacement. Neyman allocation (default) assigns more budget to strata with higher
        proxy variance, minimising asymptotic variance of downstream estimators. Proportional
        allocation allocates budget proportionally to stratum sizes and serves as a baseline.

        Parameters
        ----------
        y_proxy : NDArray
            Proxy labels for all samples, shape ``(n_total,)``. Must be 1-dimensional.
        groups : NDArray
            Stratum identifiers for all samples, shape ``(n_total,)``.
            Must be 1-dimensional with same length as y_proxy.
        n_samples : int
            Target number of annotations. Must be positive. Mandatory.
        strategy : str, optional
            Allocation strategy: "neyman" (default) or "proportional".
            "neyman": assigns more budget to higher-variance strata.
            "proportional": allocates proportionally to stratum sizes.
        random_seed : int or None, optional
            Random seed for reproducible sampling. Defaults to None (non-deterministic).

        Returns
        -------
        NDArray
            Selection indicators of shape ``(n_total,)``: 1 if the sample was selected
            for annotation, 0 otherwise.

        Raises
        ------
        ValueError
            - If ``strategy`` is not a recognized allocation strategy.
            - If ``n_samples`` is not a strictly positive integer.
            - If ``n_samples`` is too low and results in zero allocations for some stratum.
            - If ``n_samples`` exceeds the total number of samples in the input.
        """
        self._validate(y_proxy, groups, n_samples, strategy)

        if strategy == "proportional":
            allocation = self._proportional_allocation(groups, n_samples)
        else:
            allocation = self._neyman_allocation(y_proxy, groups, n_samples)

        for stratum_id, n_h in allocation.items():
            if n_h < 2:
                raise ValueError(
                    f"Stratum '{stratum_id}' has fewer than two allocations. All strata must receive at least "
                    f"two annotation slots. Consider increasing 'n_samples' or using bigger strata."
                )

            stratum_mask = groups == stratum_id
            stratum_size = stratum_mask.sum()

            if n_h > stratum_size - 2:
                raise ValueError(
                    f"Stratum '{stratum_id}' has been over-allocated. Consider using proportional sampling."
                )

        rng = np.random.default_rng(random_seed)

        xi = np.zeros(len(y_proxy), dtype=int)

        for stratum_id in np.unique(groups):
            stratum_indices = np.flatnonzero(groups == stratum_id)
            n_h = allocation[stratum_id]
            selected_samples = rng.choice(stratum_indices, size=n_h, replace=False)
            xi[selected_samples] = 1

        return xi
