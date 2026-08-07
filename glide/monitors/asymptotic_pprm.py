import numpy as np
from numpy.typing import NDArray

from glide.engines.ppi import PPIDataset, PPIMeanEngine
from glide.mean_monitoring_results import PredictionPoweredMeanMonitoringResult
from glide.monitors.base import AsymptoticRM


class AsymptoticPPRM(AsymptoticRM[PPIDataset, float]):
    """Anytime-valid drift monitor leveraging the variance of each batch estimate.

    It computes a per-batch Prediction-Powered Inference (PPI) estimate: a small set
    of true labels and a large set of proxy labels, combined with a power-tuning
    parameter fitted on the batches that strictly precede it (predictable, as
    required for validity), together with the standard error of that estimate. The
    monitor tracks the running mean of the per-batch estimates against a
    user-supplied ``threshold``, through a Gaussian-mixture asymptotic confidence
    sequence whose width scales with the standard error of each batch estimate, so
    genuine drifts get flagged early. The false-alarm guarantee is asymptotic: each
    batch needs enough labeled and proxy samples for its PPI estimate to be
    approximately Gaussian with a consistently estimated variance.

    References
    ----------
    Waudby-Smith, Ian, David Arbour, Ritwik Sinha, Edward H. Kennedy, and Aaditya
    Ramdas. "Time-uniform central limit theory and asymptotic confidence
    sequences." The Annals of Statistics 52, no. 6 (2024): 2613-2640.

    Zhang, Guangyi, Yunlong Cai, Guanding Yu, and Osvaldo Simeone. "Prediction-powered
    risk monitoring of deployed models for detecting harmful distribution shifts."
    arXiv preprint arXiv:2602.02229 (2026).

    Podkopaev, Aleksandr, and Aaditya Ramdas. "Tracking the risk of a deployed model
    and detecting harmful distribution shifts." International Conference on Learning
    Representations (ICLR), 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.monitors import AsymptoticPPRM
    >>> pre_drift_y_true = np.array([0.0, 0.2, np.nan, np.nan])
    >>> pre_drift_y_proxy = np.array([0.0, 0.2, 0.0, 0.2])
    >>> post_drift_y_true = np.array([0.8, 1.0, np.nan, np.nan])
    >>> post_drift_y_proxy = np.array([0.8, 1.0, 0.8, 1.0])
    >>> y_true = np.hstack([pre_drift_y_true, np.tile(post_drift_y_true, 5)])
    >>> y_proxy = np.hstack([pre_drift_y_proxy, np.tile(post_drift_y_proxy, 5)])
    >>> batches = np.repeat(np.arange(6), 4)
    >>> monitor = AsymptoticPPRM()
    >>> result = monitor.detect(y_true, y_proxy, batches, higher_is_better=False, threshold=0.5)
    >>> result.drift_detected
    True
    >>> result.first_alarm_index
    2
    """

    _engine = PPIMeanEngine()

    def detect(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        batches: NDArray,
        higher_is_better: bool,
        threshold: float,
        metric_name: str = "Metric",
        confidence_level: float = 0.8,
        power_tuning: bool = True,
        tightest_at_batch: int = 10,
    ) -> PredictionPoweredMeanMonitoringResult:
        """Detect a drift of the running mean across a batched dataset.

        Splits the data by batch, computes a prediction-powered estimate and its
        standard error per batch, and builds an anytime-valid asymptotic confidence
        sequence on the running mean of those estimates. An alarm is raised at every
        batch where the sequence crosses the user-supplied ``threshold``.

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
        y_true : NDArray
            Array of labeled observations, shape ``(n_samples,)``.
            Labeled entries are finite; unlabeled entries are ``np.nan``.
        y_proxy : NDArray
            Array of proxy predictions, shape ``(n_samples,)``.
            Must be fully populated (no NaN).
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
        power_tuning : bool, optional
            If ``True`` (default), compute the power-tuning parameter of each batch on
            all previous batches (the first batch, having no predecessor, uses
            ``1.0``). If ``False``, use ``1.0`` everywhere (classic PPI).
        tightest_at_batch : int, optional
            The batch index (1-indexed) at which the confidence sequence is tuned to
            be tightest. Defaults to ``10``, which is safe to leave alone: this
            setting affects tightness only, never the false-alarm guarantee, and the
            width penalty for being off by a factor of 10 is below 10%.

        Returns
        -------
        PredictionPoweredMeanMonitoringResult
            Per-batch estimates, running means, anytime-valid confidence bounds,
            alarm flags, and the alarm threshold, all in the original metric
            orientation.

        Raises
        ------
        ValueError
            - If ``batches`` is empty.
            - If ``y_true``, ``y_proxy`` and ``batches`` have different lengths.
            - If ``batches`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``confidence_level`` is not in ``(0.5, 1)``.
            - If any proxy value is NaN.
            - If batches are interleaved rather than grouped into contiguous blocks.
            - If any batch has fewer than 2 labeled or fewer than 2 unlabeled samples.
            - If proxy values are constant across a prefix set of batches (with ``power_tuning=True``).
            - If ``tightest_at_batch`` is not a positive integer.
            - If the accumulated variance of the batch estimates up to ``tightest_at_batch`` is zero.
        """
        batch_codes, batch_mean_estimates, confidence_sequence = self._detect(
            fields=[y_true, y_proxy],
            field_names=["y_true", "y_proxy"],
            batches=batches,
            higher_is_better=higher_is_better,
            confidence_level=confidence_level,
            tightest_at_batch=tightest_at_batch,
            power_tuning=power_tuning,
        )
        labeled_mask = ~np.isnan(y_true)
        batch_n_true = np.bincount(batch_codes[labeled_mask], minlength=len(batch_mean_estimates))
        batch_n_proxy = np.bincount(batch_codes, minlength=len(batch_mean_estimates))
        result = PredictionPoweredMeanMonitoringResult(
            metric_name=metric_name,
            monitor_name=self.__class__.__name__,
            higher_is_better=higher_is_better,
            alarm_threshold=threshold,
            confidence_level=confidence_level,
            batch_mean_estimates=batch_mean_estimates,
            confidence_sequence=confidence_sequence,
            batch_n_true=batch_n_true,
            batch_n_proxy=batch_n_proxy,
        )
        return result
