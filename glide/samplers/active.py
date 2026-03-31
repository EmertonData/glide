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

    Observations with **exactly zero uncertainty** are rejected outright: a zero
    uncertainty would imply an infinite drawing weight, making normalisation
    ill-defined.

    Examples
    --------
    >>> from glide.core.dataset import Dataset
    >>> from glide.samplers.active import ActiveSampler
    >>> dataset = Dataset([
    ...     {"score": 0.9, "uncertainty": 0.1},
    ...     {"score": 0.5, "uncertainty": 0.5},
    ... ])
    >>> sampler = ActiveSampler()
    >>> result = sampler.sample(dataset, uncertainty_field="uncertainty", budget=1, seed=0)
    >>> all("pi" in record and "xi" in record for record in result)
    True
    """

    def _preprocess(self, dataset: Dataset, uncertainty_field: str) -> NDArray:
        """Extract and validate uncertainty values from the dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset containing records with an ``uncertainty_field`` column.
        uncertainty_field : str
            Name of the column holding the uncertainty scores.

        Returns
        -------
        NDArray
            1D array of uncertainty values of shape ``(n,)``.

        Raises
        ------
        ValueError
            If any uncertainty value is exactly zero, or if ``uncertainty_field``
            is not present in any record.

        Examples
        --------
        >>> from glide.core.dataset import Dataset
        >>> from glide.samplers.active import ActiveSampler
        >>> dataset = Dataset([{"uncertainty": 0.2}, {"uncertainty": 0.8}])
        >>> sampler = ActiveSampler()
        >>> sampler._preprocess(dataset, "uncertainty")
        array([0.2, 0.8])
        """
        uncertainties = dataset[uncertainty_field]
        if np.any(uncertainties == 0.0):
            raise ValueError(
                f"All uncertainty values must be strictly positive; "
                f"got exactly zero in field '{uncertainty_field}'. "
                "Zero uncertainty implies infinite drawing weight and cannot be normalised."
            )
        return uncertainties

    def sample(
        self,
        dataset: Dataset,
        uncertainty_field: str,
        budget: int,
        seed: int,
        pi_field: str = "pi",
        xi_field: str = "xi",
    ) -> Dataset:
        """Sample observations with probability proportional to uncertainty.

        Each observation receives a drawing probability π_i proportional to
        ``1 / uncertainty_i``, normalised so that the raw probabilities sum to
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
            exactly zero values.
        budget : int
            Expected total number of annotations to collect. Must be a strictly
            positive integer.
        seed : int
            Random seed passed to ``numpy.random.default_rng`` for reproducibility.
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
            If ``budget`` is not a strictly positive integer, if any uncertainty
            value is exactly zero, or if ``uncertainty_field`` is not present in
            any record.

        Examples
        --------
        >>> from glide.core.dataset import Dataset
        >>> from glide.samplers.active import ActiveSampler
        >>> dataset = Dataset([
        ...     {"score": 0.9, "uncertainty": 0.1},
        ...     {"score": 0.5, "uncertainty": 0.5},
        ... ])
        >>> sampler = ActiveSampler()
        >>> result = sampler.sample(dataset, uncertainty_field="uncertainty", budget=1, seed=0)
        >>> all("pi" in record and "xi" in record for record in result)
        True
        >>> all(0 < record["pi"] <= 1 for record in result)
        True
        """
        if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
            raise ValueError(f"'budget' must be a strictly positive integer; got {budget!r}.")

        uncertainties = self._preprocess(dataset, uncertainty_field)
        rng = np.random.default_rng(seed)

        raw_weights = 1.0 / uncertainties
        drawing_probabilities = raw_weights / raw_weights.sum() * budget
        # Cap at 1: a Bernoulli probability cannot exceed 1.
        clipped_probabilities = np.minimum(drawing_probabilities, 1.0)

        indicators = rng.binomial(n=1, p=clipped_probabilities).astype(float)

        enriched_records = [
            {**record, pi_field: float(pi), xi_field: float(xi)}
            for record, pi, xi in zip(dataset.records, clipped_probabilities, indicators)
        ]
        result = Dataset(enriched_records)
        return result
