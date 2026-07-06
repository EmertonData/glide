from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.confidence_sequences.empirical_bernstein import _compute_empirical_bernstein_bounds
from glide.core.validation import _validate_bounds, _validate_equal_lengths, _validate_has_no_nan, _validate_non_empty
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult
from glide.monitors.core import _scale_from_unit_risk, _scale_to_unit_risk, _unique_ordered_batches


class ClassicalMeanMonitor:
    """Anytime-valid drift monitor over a batched dataset of labels.

    It uses the plain sample mean per batch. Unlabeled entries are passed as
    ``np.nan`` and dropped per batch. To be used on accumulated non-overlapping
    production batches by calling :meth:`detect` on the whole accumulated dataset at
    any time. Data is passed oldest batch first, and every batch is monitored. The
    monitor computes a sample-mean estimate per batch, builds an anytime-valid
    confidence sequence on the running mean of those estimates, and raises a drift
    alarm when it crosses a user-supplied ``threshold`` (the worst tolerable metric value).
    Because the bounds are valid at all times simultaneously, :meth:`detect` may be called
    after every new batch without inflating the false-alarm probability.

    References
    ----------
    Howard, Steven R., Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. "Time-uniform,
    nonparametric, nonasymptotic confidence sequences." The Annals of Statistics 49,
    no. 2 (2021): 1055-1080.

    Podkopaev, Aleksandr, and Aaditya Ramdas. "Tracking the risk of a deployed model
    and detecting harmful distribution shifts." International Conference on Learning
    Representations (ICLR), 2022.

    Waudby-Smith, Ian, and Aaditya Ramdas. "Estimating means of bounded random
    variables by betting." Journal of the Royal Statistical Society Series B:
    Statistical Methodology 86, no. 1 (2024): 1-27.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.monitors import ClassicalMeanMonitor
    >>> pre_drift_batch = np.array([0.0, 0.2, np.nan, np.nan])
    >>> post_drift_batch = np.array([0.8, 1.0, np.nan, np.nan])
    >>> y = np.hstack([pre_drift_batch, np.tile(post_drift_batch, 50)])
    >>> batches = np.repeat(np.arange(51), 4)
    >>> monitor = ClassicalMeanMonitor()
    >>> result = monitor.detect(y, batches, higher_is_better=False, threshold=0.5)
    >>> result.drift_detected
    True
    >>> result.first_alarm_index
    11
    """

    def _preprocess(
        self,
        y: NDArray,
        batches: NDArray,
        higher_is_better: bool,
        threshold: float,
        confidence_level: float,
        metric_lower_bound: float,
        metric_upper_bound: float,
    ) -> Tuple[NDArray, float, NDArray, NDArray]:
        _validate_non_empty(y, "y")
        _validate_equal_lengths(y, batches, names=["y", "batches"])
        _validate_has_no_nan(batches, "batches")
        _validate_bounds(
            confidence_level, "confidence_level", lower=0, upper=1, left_inclusive=False, right_inclusive=False
        )
        _validate_bounds(
            metric_lower_bound,
            "metric_lower_bound",
            upper=metric_upper_bound,
            right_inclusive=False,
            error_message=(
                f"'metric_lower_bound' must be strictly smaller than 'metric_upper_bound'; "
                f"got {metric_lower_bound!r} and {metric_upper_bound!r}."
            ),
        )
        _validate_bounds(
            threshold,
            "threshold",
            lower=metric_lower_bound,
            upper=metric_upper_bound,
            error_message=(
                f"'threshold' must lie between 'metric_lower_bound'={metric_lower_bound!r} and "
                f"'metric_upper_bound'={metric_upper_bound!r}."
            ),
        )
        labeled_mask = ~np.isnan(y)
        labeled_values = y[labeled_mask]
        _validate_bounds(
            labeled_values,
            "y",
            lower=metric_lower_bound,
            upper=metric_upper_bound,
            error_message=(
                f"'y' values must lie between 'metric_lower_bound'={metric_lower_bound!r} and "
                f"'metric_upper_bound'={metric_upper_bound!r}; got values in "
                f"[{labeled_values.min()!r}, {labeled_values.max()!r}]."
            ),
        )
        batches_labeled = batches[labeled_mask]
        batch_identifiers, batch_codes = _unique_ordered_batches(batches_labeled)
        batch_n = np.bincount(batch_codes)
        worst_batch_position = np.argmin(batch_n)
        _validate_bounds(
            batch_n[worst_batch_position],
            "y",
            lower=2,
            error_message=(
                f"'y' must have at least 2 non-NaN values per batch; got {batch_n[worst_batch_position]} "
                f"in batch '{batch_identifiers[worst_batch_position]}'."
            ),
        )
        risk_y = _scale_to_unit_risk(labeled_values, metric_lower_bound, metric_upper_bound, higher_is_better)
        risk_threshold = _scale_to_unit_risk(threshold, metric_lower_bound, metric_upper_bound, higher_is_better)
        return risk_y, risk_threshold, batch_codes, batch_n

    def _postprocess(
        self,
        risk_running_means: NDArray,
        risk_confidence_bounds: NDArray,
        risk_batch_mean_estimates: NDArray,
        higher_is_better: bool,
        metric_lower_bound: float,
        metric_upper_bound: float,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        running_means = _scale_from_unit_risk(
            risk_running_means, metric_lower_bound, metric_upper_bound, higher_is_better
        )
        confidence_bounds = _scale_from_unit_risk(
            risk_confidence_bounds, metric_lower_bound, metric_upper_bound, higher_is_better
        )
        batch_mean_estimates = _scale_from_unit_risk(
            risk_batch_mean_estimates, metric_lower_bound, metric_upper_bound, higher_is_better
        )
        return running_means, confidence_bounds, batch_mean_estimates

    def detect(
        self,
        y: NDArray,
        batches: NDArray,
        higher_is_better: bool,
        threshold: float,
        metric_name: str = "Metric",
        confidence_level: float = 0.8,
        metric_lower_bound: float = 0.0,
        metric_upper_bound: float = 1.0,
    ) -> ClassicalMeanMonitoringResult:
        """Detect a drift of the running mean across a batched dataset.

        Splits the data by batch, computes a sample-mean estimate per batch, and
        builds an anytime-valid empirical-Bernstein confidence sequence on the
        running mean of those estimates. An alarm is raised at every batch where the
        sequence crosses the user-supplied ``threshold``.

        Rows must be ordered oldest batch first and grouped into contiguous blocks;
        identifier values are not compared, so any label type works. Batches must be
        non-overlapping (no shared samples), and successive calls must be made on
        growing histories of the same data; passing the full accumulated dataset at
        every call makes the anytime-valid guarantee hold jointly over all calls.
        Alternatively the data may be restricted to the most recent batches, in which
        case the guarantee holds within each restriction but not across the moving
        history as a whole.

        Parameters
        ----------
        y : NDArray
            Array of observations, shape ``(n_samples,)``. Unlabeled entries are
            ``np.nan`` and are dropped per batch; every batch must keep at least 2
            labeled entries.
        batches : NDArray
            Array of batch identifiers, shape ``(n_samples,)``. Rows must be ordered
            oldest batch first and grouped into contiguous blocks. Identifier values
            are not compared, so any hashable label type works (integers, dates,
            free-form strings).
        higher_is_better : bool
            ``False`` when the metric is a risk (drift means the metric increased),
            ``True`` when it is a performance (drift means the metric decreased).
        threshold : float
            The metric value the running mean is monitored against, in metric units:
            the worst level the user is willing to tolerate. An alarm fires once the
            anytime-valid bound proves the running metric has crossed it (the running
            risk exceeds it for a risk, the running performance falls below it for a
            performance). Must lie within ``[metric_lower_bound, metric_upper_bound]``.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            How confident each alarm should be. At the default ``0.8`` the monitor
            raises "80% confident" alarms: under no drift it has at most a 20% chance
            of raising a false alarm. Raising this value demands more evidence before
            alarming, so alarms become more trustworthy but arrive later; lowering it
            detects sooner at the cost of more false alarms.
        metric_lower_bound : float, optional
            Known lower bound of the metric. Defaults to ``0.0``.
        metric_upper_bound : float, optional
            Known upper bound of the metric. Defaults to ``1.0``.

        Returns
        -------
        ClassicalMeanMonitoringResult
            Per-batch estimates, running means, anytime-valid confidence bounds,
            alarm flags, and the alarm threshold, all in the original metric
            orientation.

        Raises
        ------
        ValueError
            - If ``y`` is empty.
            - If ``y`` and ``batches`` have different lengths.
            - If ``batches`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``confidence_level`` is not in (0, 1).
            - If ``metric_lower_bound >= metric_upper_bound``.
            - If ``threshold`` falls outside ``[metric_lower_bound, metric_upper_bound]``.
            - If any labeled value falls outside ``[metric_lower_bound, metric_upper_bound]``.
            - If batches are interleaved rather than grouped into contiguous blocks.
            - If any batch has fewer than 2 labeled (non-NaN) samples.
        """
        risk_y, risk_threshold, batch_codes, batch_n = self._preprocess(
            y, batches, higher_is_better, threshold, confidence_level, metric_lower_bound, metric_upper_bound
        )
        batch_sums = np.bincount(batch_codes, weights=risk_y)
        risk_batch_mean_estimates = batch_sums / batch_n

        miscoverage = 1.0 - confidence_level
        risk_running_means, risk_confidence_bounds = _compute_empirical_bernstein_bounds(
            risk_batch_mean_estimates, risk_threshold, miscoverage
        )

        running_means, confidence_bounds, batch_mean_estimates = self._postprocess(
            risk_running_means,
            risk_confidence_bounds,
            risk_batch_mean_estimates,
            higher_is_better,
            metric_lower_bound,
            metric_upper_bound,
        )
        confidence_sequence = EmpiricalBernsteinConfidenceSequence(
            running_mean_estimates=running_means,
            confidence_bounds=confidence_bounds,
        )
        result = ClassicalMeanMonitoringResult(
            metric_name=metric_name,
            monitor_name=self.__class__.__name__,
            higher_is_better=higher_is_better,
            alarm_threshold=threshold,
            confidence_level=confidence_level,
            batch_mean_estimates=batch_mean_estimates,
            confidence_sequence=confidence_sequence,
            batch_n=batch_n,
        )
        return result
