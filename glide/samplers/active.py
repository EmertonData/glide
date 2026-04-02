from typing import Optional

import numpy as np
from numpy.typing import NDArray

from glide.core.dataset import Dataset


class ActiveSampler:
    """Sampler that draws elements from a dataset with probabilities based on an
    uncertainty field.

    Implements active sampling for inference pipelines which support inverse
    probability weighting (IPW).
    Each observation is assigned a drawing probability π_i proportional to its
    uncertainty score, then independently selected via a Bernoulli trial. This
    concentrates the annotation budget on the most uncertain observations.

    Examples
    --------
    >>> from glide.core.dataset import Dataset
    >>> from glide.samplers.active import ActiveSampler
    >>> dataset = Dataset([
    ...     {"score": 0.9, "uncertainty": 0.1},
    ...     {"score": 0.5, "uncertainty": 0.4},
    ... ])
    >>> sampler = ActiveSampler()
    >>> result = sampler.sample(dataset, uncertainty_field="uncertainty", budget=1, random_seed=0)
    >>> result["pi"]
    array([0.2, 0.8])
    >>> result["xi"]
    array([0., 1.])
    >>> all("pi" in record and "xi" in record for record in result)
    True
    >>> all(0 < record["pi"] <= 1 for record in result)
    True
    """

    def _preprocess(self, dataset: Dataset, uncertainty_field: str) -> NDArray:

        uncertainties = dataset[uncertainty_field]
        if np.any(np.isnan(uncertainties)):
            raise ValueError(
                f"All uncertainty values must be finite; "
                f"got a NaN or absent value in field '{uncertainty_field}'. "
                "A missing or NaN uncertainty score cannot be used to compute sampling probabilities."
            )
        if np.any(uncertainties <= 0.0):
            raise ValueError(
                f"All uncertainty values must be strictly positive; "
                f"got a non-positive value in field '{uncertainty_field}'. "
                "An observation with zero or negative uncertainty would never be selected."
            )
        return uncertainties

    def sample(
        self,
        dataset: Dataset,
        uncertainty_field: str,
        budget: int,
        random_seed: Optional[int] = None,
        pi_field: str = "pi",
        xi_field: str = "xi",
    ) -> Dataset:
        """Sample observations with probability proportional to uncertainty.

        Each observation receives a drawing probability π_i proportional to
        ``uncertainty_i``, normalised so that the raw probabilities sum to
        ``budget`` (the expected number of selected observations). Because each
        π_i must be a valid Bernoulli probability, values are capped at 1 before
        the coin flip; the actual number of selected items is therefore a random
        variable whose expectation equals at most ``budget``.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing records with an ``uncertainty_field`` column.
        uncertainty_field : str
            Name of the column holding the uncertainty scores. Must not contain
            zero or negative values.
        budget : int
            Expected total number of annotations to collect. Must be a strictly
            positive integer and must not exceed the number of records in
            ``dataset``.
        random_seed : int or None, optional
            Random seed passed to ``numpy.random.default_rng`` for
            reproducibility. Pass ``None`` (the default) to use a
            non-deterministic random_seed.
        pi_field : str, optional
            Name of the output column for drawing probabilities. Defaults to
            ``"pi"``.
        xi_field : str, optional
            Name of the output column for Bernoulli selection indicators.
            Defaults to ``"xi"``.

        Returns
        -------
        Dataset
            Copy of the input dataset with two additional fields per record:
            ``pi_field`` (drawing probability in ``(0, 1]``) and ``xi_field``
            (1 if the observation was selected for annotation, 0 otherwise).

        Raises
        ------
        ValueError
            If ``budget`` is not a strictly positive integer, if ``budget``
            exceeds the number of records in ``dataset``, if any uncertainty
            value is NaN or absent (field missing from a record), if any value is
            zero or negative, or if ``uncertainty_field`` is not present in any
            record.
        """
        if (not isinstance(budget, (int, np.integer))) or isinstance(budget, bool) or budget <= 0:
            raise ValueError(f"'budget' must be a strictly positive integer; got {budget!r}.")
        if budget > len(dataset):
            raise ValueError(
                f"'budget' must not exceed the number of records in the dataset; "
                f"got budget={budget} but dataset has {len(dataset)} records."
            )

        uncertainties = self._preprocess(dataset, uncertainty_field)
        rng = np.random.default_rng(random_seed)

        drawing_probabilities = budget * uncertainties / uncertainties.sum()
        # Cap at 1: a Bernoulli probability cannot exceed 1.
        clipped_probabilities = np.minimum(drawing_probabilities, 1.0)

        indicators = rng.binomial(n=1, p=clipped_probabilities).astype(float)

        enriched_records = [
            {**record, pi_field: pi, xi_field: xi}
            for record, pi, xi in zip(dataset.records, clipped_probabilities, indicators)
        ]
        result = Dataset(enriched_records)
        return result
