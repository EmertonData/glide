from typing import Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.estimators import ClassicalMeanEstimator


def run_monte_carlo(
    methods: List[str],
    confidence_levels: NDArray,
    run_seed: Callable[[int], Dict],
    n_seeds: int = 500,
) -> Dict:
    """Run a Monte Carlo simulation over independent seeds.

    Parameters
    ----------
    methods : List[str]
        Names of the estimation methods to compare.
    confidence_levels : NDArray
        Confidence levels at which to evaluate interval bounds.
    run_seed : Callable[[int], Dict]
        Function that takes a seed integer and returns a dict mapping each method
        name to ``{"mean": float, "std": float, "confidence_interval": <result>}``.
        May also include ``"effective_sample_size": float`` for any method.
    n_seeds : int, optional
        Number of Monte Carlo seeds. Default is 500.

    Returns
    -------
    Dict
        Nested dict mapping each method name to
        ``{"means": NDArray, "stds": NDArray, "lower_bounds": {level: NDArray},
        "upper_bounds": {level: NDArray}, "effective_sample_sizes": NDArray}``.
        ``effective_sample_sizes`` is ``NaN`` for seeds or methods that did not
        include it.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.scientific_validation import run_monte_carlo
    >>> from glide.estimators import ClassicalMeanEstimator
    >>> def run_seed(seed):
    ...     y = np.array([0.0, 1.0])
    ...     result = ClassicalMeanEstimator().estimate(y, confidence_level=0.9)
    ...     return {"M": {"mean": result.mean, "std": result.std, "confidence_interval": result.confidence_interval}}
    >>> stats = run_monte_carlo(["M"], np.array([0.9]), run_seed, n_seeds=2)
    >>> stats["M"]["means"]
    array([0.5, 0.5])
    """
    if n_seeds <= 0:
        raise ValueError(f"'n_seeds' must be > 0; got {n_seeds!r}.")
    if not methods:
        raise ValueError("'methods' must be non-empty.")
    if not np.all((confidence_levels > 0) & (confidence_levels < 1)):
        raise ValueError(f"All 'confidence_levels' must be in (0, 1); got {confidence_levels!r}.")
    means = {method: np.zeros(n_seeds) for method in methods}
    stds = {method: np.zeros(n_seeds) for method in methods}
    lower_bounds = {method: {level: np.zeros(n_seeds) for level in confidence_levels} for method in methods}
    upper_bounds = {method: {level: np.zeros(n_seeds) for level in confidence_levels} for method in methods}
    effective_sample_sizes = {method: np.full(n_seeds, np.nan) for method in methods}

    for seed in range(n_seeds):
        estimates = run_seed(seed)
        for method in methods:
            means[method][seed] = estimates[method]["mean"]
            stds[method][seed] = estimates[method]["std"]
            confidence_interval = estimates[method]["confidence_interval"]
            for level in confidence_levels:
                confidence_interval.confidence_level = level
                lower_bounds[method][level][seed] = confidence_interval.lower_bound
                upper_bounds[method][level][seed] = confidence_interval.upper_bound
            if "effective_sample_size" in estimates[method]:
                effective_sample_sizes[method][seed] = estimates[method]["effective_sample_size"]

    stats = {
        method: {
            "means": means[method],
            "stds": stds[method],
            "lower_bounds": lower_bounds[method],
            "upper_bounds": upper_bounds[method],
            "effective_sample_sizes": effective_sample_sizes[method],
        }
        for method in methods
    }
    return stats


def compute_hits(
    stats: Dict,
    confidence_level: float,
    true_mean: float,
) -> Dict[str, NDArray]:
    """Return per-seed hit indicators for each method at a given confidence level.

    A hit is 1 when the confidence interval for that seed contains ``true_mean``,
    and 0 otherwise.

    Parameters
    ----------
    stats : Dict
        Output of :func:`run_monte_carlo`.
    confidence_level : float
        The confidence level at which to evaluate coverage.
    true_mean : float
        The ground-truth value that the intervals should cover.

    Returns
    -------
    Dict[str, NDArray]
        Dict mapping each method name to a float array of shape ``(n_seeds,)``
        with values in ``{0.0, 1.0}``.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.scientific_validation import run_monte_carlo, compute_hits
    >>> from glide.estimators import ClassicalMeanEstimator
    >>> def run_seed(seed):
    ...     y = np.array([0.0, 1.0])
    ...     result = ClassicalMeanEstimator().estimate(y, confidence_level=0.9)
    ...     return {"M": {"mean": result.mean, "std": result.std, "confidence_interval": result.confidence_interval}}
    >>> stats = run_monte_carlo(["M"], np.array([0.9]), run_seed, n_seeds=2)
    >>> hits = compute_hits(stats, confidence_level=0.9, true_mean=0.5)
    >>> hits["M"]
    array([1., 1.])
    """
    if not 0 < confidence_level < 1:
        raise ValueError(f"'confidence_level' must be in (0, 1); got {confidence_level!r}.")
    hits = {}
    for method in stats:
        if confidence_level not in stats[method]["lower_bounds"]:
            available_levels = sorted(stats[method]["lower_bounds"].keys())
            raise ValueError(
                f"'confidence_level' {confidence_level!r} was not passed to run_monte_carlo; "
                f"available levels are {available_levels!r}."
            )
        lower = stats[method]["lower_bounds"][confidence_level]
        upper = stats[method]["upper_bounds"][confidence_level]
        hits[method] = np.asarray((lower <= true_mean) & (true_mean <= upper), dtype=float)
    return hits


def coverage_with_error_bar(
    hits: NDArray,
    confidence_level: float,
) -> Tuple[float, float, float]:
    """Estimate empirical coverage and its confidence interval from hit indicators.

    Uses :class:`~glide.estimators.ClassicalMeanEstimator` on the binary hit array,
    which gives a valid confidence interval on the coverage rate via the normal
    approximation.

    Parameters
    ----------
    hits : NDArray
        Float array of per-seed hit indicators (values in ``{0.0, 1.0}``).
    confidence_level : float
        Confidence level for the error bar on the coverage estimate.

    Returns
    -------
    Tuple[float, float, float]
        ``(mean_coverage, lower_bound, upper_bound)``.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.scientific_validation import coverage_with_error_bar
    >>> hits = np.array([1.0, 1.0, 0.0, 1.0])
    >>> mean_cov, lower, upper = coverage_with_error_bar(hits, confidence_level=0.95)
    >>> float(mean_cov)
    0.75
    """
    if len(hits) == 0:
        raise ValueError("'hits' must be non-empty.")
    if not 0 < confidence_level < 1:
        raise ValueError(f"'confidence_level' must be in (0, 1); got {confidence_level!r}.")
    estimator = ClassicalMeanEstimator()
    result = estimator.estimate(hits, confidence_level=confidence_level)
    coverage = result.mean
    lower = result.confidence_interval.lower_bound
    upper = result.confidence_interval.upper_bound
    return coverage, lower, upper
