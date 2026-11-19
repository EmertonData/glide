import numpy as np
from numpy.typing import NDArray

from glide.confidence_sequences import AsymptoticConfidenceSequence
from glide.engines.ipw_classical import IPWClassicalDataset, IPWClassicalMeanEngine
from glide.mean_monitoring_results import ClassicalMeanMonitoringResult
from glide.monitors.base import AsymptoticRM


class AsymptoticIPWClassicalRM(AsymptoticRM[IPWClassicalDataset, None]):
    """Anytime-valid label-only drift monitor for non-uniformly labeled data.

    It computes a per-batch inverse-probability-weighted mean of the labeled
    values, where each labeled value is divided by the known probability with
    which it was selected for annotation, then tracks the running mean of those
    estimates against a user-supplied ``threshold``. The comparison uses a
    Gaussian-mixture asymptotic confidence sequence whose width scales with the
    standard error of each batch estimate, exactly like ``AsymptoticClassicalRM``
    does for uniformly labeled data. The false-alarm guarantee is asymptotic:
    each batch needs enough labeled samples (preferably tens) for its estimate to
    be approximately Gaussian with a consistently estimated variance. Under
    severely non-uniform probabilities, a few large inverse weights can dominate
    a batch estimate, so more labeled samples per batch are needed than under
    uniform labeling.

    References
    ----------
    Waudby-Smith, Ian, David Arbour, Ritwik Sinha, Edward H. Kennedy, and Aaditya
    Ramdas. "Time-uniform central limit theory and asymptotic confidence
    sequences." The Annals of Statistics 52, no. 6 (2024): 2613-2640.

    Zrnic, Tijana, and Emmanuel J. Candès. "Active statistical inference." In Proceedings
    of the 41st International Conference on Machine Learning, pp. 62993-63010. 2024.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.monitors import AsymptoticIPWClassicalRM
    >>> pre_drift_y = np.array([0.0, 0.2, np.nan, np.nan])
    >>> post_drift_y = np.array([0.8, 1.0, np.nan, np.nan])
    >>> y = np.hstack([pre_drift_y, np.tile(post_drift_y, 5)])
    >>> sampling_probability = np.full(24, 0.5)
    >>> batches = np.repeat(np.arange(6), 4)
    >>> monitor = AsymptoticIPWClassicalRM()
    >>> result = monitor.detect(y, sampling_probability, batches, higher_is_better=False, threshold=0.3)
    >>> result.drift_detected
    True
    >>> result.first_alarm_index
    5
    """

    _engine = IPWClassicalMeanEngine()

    def detect(
        self,
        y: NDArray,
        sampling_probability: NDArray,
        batches: NDArray,
        *,
        higher_is_better: bool,
        threshold: float,
        metric_name: str = "Metric",
        confidence_level: float = 0.8,
        tightest_at_batch: int = 10,
    ) -> ClassicalMeanMonitoringResult:
        """Detect a drift of the running mean across a batched, non-uniformly labeled dataset.

        Splits the data by batch, computes an inverse-probability-weighted mean and its
        standard error per batch, and builds an anytime-valid asymptotic confidence
        sequence on the running mean of those estimates. An alarm is raised at every
        batch where the sequence crosses the user-supplied ``threshold``. See
        ``AsymptoticClassicalRM.detect`` for the full explanation of batch ordering,
        growing-history semantics, and ``tightest_at_batch``, which all carry over
        unchanged.

        Parameters
        ----------
        y : NDArray
            Array of observations, shape ``(n_samples,)``. Unlabeled entries are
            ``np.nan``; every batch must keep at least 2 labeled entries, though tens
            are recommended in practice for the asymptotic guarantee to hold.
        sampling_probability : NDArray
            Array of pre-determined selection probabilities, shape ``(n_samples,)``,
            each in ``[0, 1]``. Entries equal to ``0`` are excluded from the
            computation. A labeled entry must have a strictly positive probability.
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
            performance).
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            How confident each alarm should be. At the default ``0.8`` the monitor
            raises "80% confident" alarms: under no drift it has at most a 20% chance
            of raising a false alarm. Raising this value demands more evidence before
            alarming, so alarms become more trustworthy but arrive later; lowering it
            detects sooner at the cost of more false alarms. Must be in ``(0.5, 1)``.
        tightest_at_batch : int, optional
            The batch index (1-indexed) at which the confidence sequence is tuned to
            be tightest. Defaults to ``10``. Only affects tightness, not validity:
            bounds stay anytime-valid at every batch regardless of this choice.

        Returns
        -------
        ClassicalMeanMonitoringResult
            Per-batch estimates, running means, anytime-valid confidence bounds,
            alarm flags, and the alarm threshold, all in the original metric
            orientation. ``batch_n`` counts the labeled samples of each batch.

        Raises
        ------
        ValueError
            - If ``batches`` is empty.
            - If ``y``, ``sampling_probability`` and ``batches`` have different lengths.
            - If ``batches`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``confidence_level`` is not in ``(0.5, 1)``.
            - If batches are interleaved rather than grouped into contiguous blocks.
            - If any probability is outside ``[0, 1]``.
            - If a labeled entry has probability ``0``, or an unlabeled entry has probability ``1``.
            - If any batch has fewer than 2 labeled (non-NaN) samples.
            - If ``tightest_at_batch`` is not a positive integer.
            - If the accumulated variance of the batch estimates up to ``tightest_at_batch`` is zero.
        """
        batch_codes, batch_mean_estimates, running_means, confidence_bounds = self._detect(
            label_fields={"y": y},
            batches=batches,
            identifier_fields={"sampling_probability": sampling_probability},
            higher_is_better=higher_is_better,
            confidence_level=confidence_level,
            tightest_at_batch=tightest_at_batch,
            power_tuning=False,
        )
        confidence_sequence = AsymptoticConfidenceSequence(
            running_mean_estimates=running_means, confidence_bounds=confidence_bounds
        )
        labeled_mask = ~np.isnan(y)
        batch_n = np.bincount(batch_codes[labeled_mask])
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
