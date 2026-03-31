from typing import Dict, Hashable, Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.core.dataset import Dataset


class StratifiedSampler:
    """Sampler for optimal per-stratum annotation budget allocation.

    This class implements stratified sampling strategies that determine how many records
    to annotate from each stratum, given a fixed annotation budget and proxy labels for
    all records. It supports two allocation strategies: proportional (baseline) and
    Neyman allocation (optimal, reduces CI width under heterogeneous proxy variance).

    The sampler uses proxy variance within each stratum to guide allocation. Neyman
    allocation assigns more budget to strata with higher proxy variance, minimising
    the asymptotic variance of downstream estimators (e.g., StratifiedPPIMeanEstimator,
    ASIMeanEstimator). Proportional allocation serves as a simple baseline.

    Both allocators guarantee exact budget compliance via largest-remainder rounding
    (Hamilton's method).

    References
    ----------
    Fogliato, Riccardo, Pratik Patil, Mathew Monfort, and Pietro Perona.
    "A framework for efficient model evaluation through stratification, sampling,
    and estimation." In European Conference on Computer Vision, pp. 140-158.
    Cham: Springer Nature Switzerland, 2024. https://arxiv.org/abs/2406.07320

    Examples
    --------
    >>> from glide.core.dataset import Dataset
    >>> from glide.samplers.stratified import StratifiedSampler
    >>> dataset = Dataset([
    ...     {"group": "A", "y_proxy": 0.9},
    ...     {"group": "A", "y_proxy": 0.8},
    ...     {"group": "B", "y_proxy": 0.2},
    ...     {"group": "B", "y_proxy": 0.3},
    ... ])
    >>> sampler = StratifiedSampler()
    >>> allocation = sampler.allocate_budget(
    ...     dataset, groups_field="group", y_proxy_field="y_proxy", budget=2
    ... )
    >>> allocation  # doctest: +SKIP
    {'A': 1, 'B': 1}
    >>> allocation_proportional = sampler.allocate_budget(
    ...     dataset,
    ...     groups_field="group",
    ...     y_proxy_field="y_proxy",
    ...     budget=2,
    ...     strategy="proportional"
    ... )
    >>> allocation_proportional  # doctest: +SKIP
    {'A': 1, 'B': 1}
    """

    def _preprocess(
        self,
        dataset: Dataset,
        groups_field: str,
        y_proxy_field: str,
    ) -> Tuple[NDArray, NDArray]:
        """Extract proxy labels and stratum identifiers from dataset.

        Parameters
        ----------
        dataset : Dataset
            Input dataset with records.
        groups_field : str
            Name of the field holding stratum identifiers.
        y_proxy_field : str
            Name of the field holding proxy labels.

        Returns
        -------
        y_proxy : NDArray
            1D array of proxy labels, shape (n,).
        groups : NDArray
            1D array of stratum identifiers, shape (n,).

        Raises
        ------
        KeyError
            If groups_field is missing from a record.
        ValueError
            If y_proxy_field is not present in any record.
        """
        data = dataset.to_numpy(fields=[y_proxy_field])
        y_proxy = data[:, 0]
        groups = np.array([record[groups_field] for record in dataset])
        return y_proxy, groups

    def _apply_largest_remainder_rounding(
        self,
        raw_allocation: Dict[Hashable, float],
        budget: int,
    ) -> Dict[Hashable, int]:
        """Apply largest-remainder rounding to ensure exact budget sum.

        Assigns the floor value to each stratum, then distributes remaining slots
        one-by-one to strata with the largest fractional parts.

        Parameters
        ----------
        raw_allocation : Dict[Hashable, float]
            Dictionary mapping stratum_id to raw (float) allocation.
        budget : int
            Target budget sum.

        Returns
        -------
        allocation : Dict[Hashable, int]
            Dictionary mapping stratum_id to integer allocation that sums to budget.
        """
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
        for i in range(remaining_slots):
            stratum_id = sorted_strata[i][0]
            allocation[stratum_id] += 1

        return allocation

    def _proportional_allocation(
        self,
        y_proxy: NDArray,
        groups: NDArray,
        budget: int,
    ) -> Dict[Hashable, int]:
        """Allocate budget proportionally to stratum sizes.

        Parameters
        ----------
        y_proxy : NDArray
            Proxy labels, shape (n,).
        groups : NDArray
            Stratum identifiers, shape (n,).
        budget : int
            Total annotation budget.

        Returns
        -------
        allocation : Dict[Hashable, int]
            Mapping from stratum_id to per-stratum budget.
        """
        unique_strata = np.unique(groups)
        total_size = len(groups)

        raw_allocation = {}
        for stratum_id in unique_strata:
            stratum_size = (groups == stratum_id).sum()
            raw_allocation[stratum_id] = budget * stratum_size / total_size

        allocation = self._apply_largest_remainder_rounding(raw_allocation, budget)

        for stratum_id in allocation:
            stratum_size = (groups == stratum_id).sum()
            allocation[stratum_id] = max(1, min(allocation[stratum_id], stratum_size))

        return allocation

    def _neyman_allocation(
        self,
        y_proxy: NDArray,
        groups: NDArray,
        budget: int,
    ) -> Dict[Hashable, int]:
        """Allocate budget via Neyman allocation (optimal for heterogeneous variance).

        Allocates more budget to strata with higher proxy variance. If all strata have
        zero variance, falls back to proportional allocation.

        Parameters
        ----------
        y_proxy : NDArray
            Proxy labels, shape (n,).
        groups : NDArray
            Stratum identifiers, shape (n,).
        budget : int
            Total annotation budget.

        Returns
        -------
        allocation : Dict[Hashable, int]
            Mapping from stratum_id to per-stratum budget.
        """
        unique_strata = np.unique(groups)

        weights = {}
        for stratum_id in unique_strata:
            stratum_mask = groups == stratum_id
            stratum_size = stratum_mask.sum()
            stratum_y_proxy = y_proxy[stratum_mask]
            stratum_variance = np.std(stratum_y_proxy, ddof=1)
            weight = stratum_size * stratum_variance
            weights[stratum_id] = weight

        total_weight = sum(weights.values())

        if total_weight == 0:
            return self._proportional_allocation(y_proxy, groups, budget)

        raw_allocation = {}
        for stratum_id in unique_strata:
            raw_allocation[stratum_id] = budget * weights[stratum_id] / total_weight

        allocation = self._apply_largest_remainder_rounding(raw_allocation, budget)

        for stratum_id in allocation:
            stratum_size = (groups == stratum_id).sum()
            allocation[stratum_id] = max(1, min(allocation[stratum_id], stratum_size))

        return allocation

    def allocate_budget(
        self,
        dataset: Dataset,
        groups_field: str,
        y_proxy_field: str,
        budget: int,
        strategy: Literal["proportional", "neyman"] = "neyman",
    ) -> Dict[Hashable, int]:
        """Allocate annotation budget across strata.

        Parameters
        ----------
        dataset : Dataset
            Dataset with all records and proxy labels.
        groups_field : str
            Field name holding stratum identifiers. Mandatory.
        y_proxy_field : str
            Field name holding proxy labels. Mandatory.
        budget : int
            Total number of annotations to collect. Must be positive. Mandatory.
        strategy : Literal["proportional", "neyman"], optional
            Allocation strategy. "neyman" (default) allocates more budget to strata
            with higher proxy variance; "proportional" allocates proportionally to
            stratum sizes.

        Returns
        -------
        allocation : Dict[Hashable, int]
            Mapping from stratum_id to per-stratum allocation. Sum equals budget exactly.

        Raises
        ------
        KeyError
            If groups_field is missing from a record.
        ValueError
            If y_proxy_field is not present in any record, or if strategy is unknown.
        """
        y_proxy, groups = self._preprocess(dataset, groups_field, y_proxy_field)

        if strategy == "proportional":
            result = self._proportional_allocation(y_proxy, groups, budget)
        elif strategy == "neyman":
            result = self._neyman_allocation(y_proxy, groups, budget)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Expected 'proportional' or 'neyman'."
            )

        return result
