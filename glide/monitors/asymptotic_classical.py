from numpy.typing import NDArray

from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.confidence_sequences.asymptotic import _compute_asymptotic_bounds
from glide.core.validation import _validate_bounds
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult
from glide.monitors.classical_core import _compute_batch_estimates, _preprocess
from glide.monitors.core import _postprocess


class AsymptoticClassicalMeanMonitor:
    """Anytime-valid label-only drift monitor leveraging each batch estimate's
    standard deviation.

    It computes a per-batch sample mean of the labeled values, then tracks the
    running mean of those estimates against a user-supplied ``threshold``. The
    comparison uses a Gaussian-mixture asymptotic confidence sequence whose
    width scales with the standard error of each batch mean, so genuine drifts
    get flagged early. The false-alarm guarantee is asymptotic: each batch
    needs enough labeled samples (preferably tens) for its sample mean to be
    approximately Gaussian with a consistently estimated variance.

    References
    ----------
    Waudby-Smith, Ian, David Arbour, Ritwik Sinha, Edward H. Kennedy, and Aaditya
    Ramdas. "Time-uniform central limit theory and asymptotic confidence
    sequences." The Annals of Statistics 52, no. 6 (2024): 2613-2640.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.monitors import AsymptoticClassicalMeanMonitor
    >>> pre_drift_batch = np.array([0.0, 0.2, np.nan, np.nan])
    >>> post_drift_batch = np.array([0.8, 1.0, np.nan, np.nan])
    >>> y = np.hstack([pre_drift_batch, np.tile(post_drift_batch, 5)])
    >>> batches = np.repeat(np.arange(6), 4)
    >>> monitor = AsymptoticClassicalMeanMonitor()
    >>> result = monitor.detect(y, batches, higher_is_better=False, threshold=0.5)
    >>> result.drift_detected
    True
    >>> result.first_alarm_index
    3
    """

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
        tightest_at_batch: int = 10,
    ) -> ClassicalMeanMonitoringResult:
        """Detect a drift of the running mean across a batched dataset.

        Splits the data by batch, computes a sample-mean estimate and its standard
        error per batch, and builds an anytime-valid asymptotic confidence sequence
        on the running mean of those estimates. An alarm is raised at every batch
        where the sequence crosses the user-supplied ``threshold``.

        Rows must be ordered oldest batch first and grouped into contiguous blocks;
        identifier values are not compared, so any label type works. Batches must be
        non-overlapping (no shared samples), and successive calls must be made on
        growing histories of the same data; passing the full accumulated dataset at
        every call makes the anytime-valid guarantee hold jointly over all calls.
        Alternatively the data may be restricted to the most recent batches, in which
        case the guarantee holds within each restriction but not across the moving
        history as a whole. When the history has fewer batches than ``tightest_at_batch``,
        the tuning target is set to the last batch. Prefix-consistency only holds between
        calls using the same ``tightest_at_batch`` value and with histories containing
        at least ``tightest_at_batch`` batches.

        Parameters
        ----------
        y : NDArray
            Array of observations, shape ``(n_samples,)``. Unlabeled entries are
            ``np.nan`` and are dropped per batch; every batch must keep at least 2
            labeled entries, though tens are recommended in practice for the
            asymptotic guarantee to hold.
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
            detects sooner at the cost of more false alarms. Must be in ``(0.5, 1)``.
        metric_lower_bound : float, optional
            Known lower bound of the metric. Defaults to ``0.0``.
        metric_upper_bound : float, optional
            Known upper bound of the metric. Defaults to ``1.0``.
        tightest_at_batch : int, optional
            The batch index (1-indexed) at which the confidence sequence is tuned to
            be tightest. Defaults to ``10``. Only affects tightness, not validity:
            bounds stay anytime-valid at every batch regardless of this choice.

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
            - If ``confidence_level`` is not in ``(0.5, 1)``.
            - If ``metric_lower_bound >= metric_upper_bound``.
            - If ``threshold`` falls outside ``[metric_lower_bound, metric_upper_bound]``.
            - If any labeled value falls outside ``[metric_lower_bound, metric_upper_bound]``.
            - If batches are interleaved rather than grouped into contiguous blocks.
            - If any batch has fewer than 2 labeled (non-NaN) samples.
            - If the accumulated variance of the batch estimates up to ``tightest_at_batch`` is zero.
        """
        _validate_bounds(
            confidence_level,
            "confidence_level",
            lower=0.5,
            upper=1,
            left_inclusive=False,
            right_inclusive=False,
            error_message=(
                f"'confidence_level' must be in (0.5, 1) for the asymptotic monitor; got {confidence_level!r}."
            ),
        )
        risk_y, _, batch_codes, batch_n = _preprocess(
            y, batches, higher_is_better, threshold, confidence_level, metric_lower_bound, metric_upper_bound
        )
        batch_mean_estimates, batch_std_estimates = _compute_batch_estimates(risk_y, batch_codes)
        miscoverage = 1.0 - confidence_level
        risk_running_means, risk_lower_bounds = _compute_asymptotic_bounds(
            batch_mean_estimates, batch_std_estimates, miscoverage, tightest_at_batch
        )
        running_means, confidence_bounds, metric_batch_estimates = _postprocess(
            risk_running_means,
            risk_lower_bounds,
            batch_mean_estimates,
            higher_is_better,
            metric_lower_bound,
            metric_upper_bound,
        )
        confidence_sequence = AsymptoticConfidenceSequence(
            running_mean_estimates=running_means,
            confidence_bounds=confidence_bounds,
        )
        result = ClassicalMeanMonitoringResult(
            metric_name=metric_name,
            monitor_name=self.__class__.__name__,
            higher_is_better=higher_is_better,
            alarm_threshold=threshold,
            confidence_level=confidence_level,
            batch_mean_estimates=metric_batch_estimates,
            confidence_sequence=confidence_sequence,
            batch_n=batch_n,
        )
        return result
