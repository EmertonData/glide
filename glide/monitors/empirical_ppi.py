import numpy as np
from numpy.typing import NDArray

from glide.confidence_sequences import EmpiricalBernsteinConfidenceSequence
from glide.confidence_sequences.empirical_bernstein import _compute_empirical_bernstein_bounds
from glide.mean_monitoring_results import PredictionPoweredMeanMonitoringResult
from glide.monitors.core import _postprocess
from glide.monitors.ppi_core import _compute_batch_estimates, _preprocess


class EmpiricalPPIMeanMonitor:
    """Anytime-valid drift monitor over a batched dataset of true and proxy labels.

    It uses a per-batch Prediction-Powered Inference (PPI) estimate: a small set of
    true labels and a large set of proxy labels, combined with a power-tuning
    parameter fitted on the batches that strictly precede it (predictable, as
    required for validity). The monitor builds an anytime-valid confidence sequence
    on the running mean of the per-batch PPI estimates, and raises a drift alarm
    when it crosses a user-supplied ``threshold`` (the worst tolerable metric
    value). Because the bounds are valid at all times simultaneously, :meth:`detect`
    may be called after every new batch without inflating the false-alarm
    probability.

    References
    ----------
    Zhang, Guangyi, Yunlong Cai, Guanding Yu, and Osvaldo Simeone. "Prediction-powered
    risk monitoring of deployed models for detecting harmful distribution shifts."
    arXiv preprint arXiv:2602.02229 (2026).

    Howard, Steven R., Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. "Time-uniform,
    nonparametric, nonasymptotic confidence sequences." The Annals of Statistics 49,
    no. 2 (2021): 1055-1080.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.monitors import EmpiricalPPIMeanMonitor
    >>> pre_drift_y_true = np.array([0.0, 0.2, np.nan, np.nan])
    >>> pre_drift_y_proxy = np.array([0.0, 0.2, 0.0, 0.2])
    >>> post_drift_y_true = np.array([0.8, 1.0, np.nan, np.nan])
    >>> post_drift_y_proxy = np.array([0.8, 1.0, 0.8, 1.0])
    >>> y_true = np.hstack([pre_drift_y_true, np.tile(post_drift_y_true, 15)])
    >>> y_proxy = np.hstack([pre_drift_y_proxy, np.tile(post_drift_y_proxy, 15)])
    >>> batches = np.repeat(np.arange(16), 4)
    >>> monitor = EmpiricalPPIMeanMonitor()
    >>> result = monitor.detect(y_true, y_proxy, batches, higher_is_better=False, threshold=0.5)
    >>> result.drift_detected
    True
    >>> result.first_alarm_index
    11
    """

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
        metric_lower_bound: float = 0.0,
        metric_upper_bound: float = 1.0,
    ) -> PredictionPoweredMeanMonitoringResult:
        """Detect a drift of the running mean across a batched dataset.

        Splits the data by batch, computes a prediction-powered estimate per batch,
        and builds an anytime-valid empirical-Bernstein confidence sequence on the
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
            performance). Must lie within ``[metric_lower_bound, metric_upper_bound]``.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            How confident each alarm should be. At the default ``0.8`` the monitor
            raises "80% confident" alarms: under no drift it has at most a 20% chance
            of raising a false alarm. Raising this value demands more evidence before
            alarming, so alarms become more trustworthy but arrive later; lowering it
            detects sooner at the cost of more false alarms.
        power_tuning : bool, optional
            If ``True`` (default), compute the power-tuning parameter of each batch on
            all previous batches (the first batch, having no predecessor, uses
            ``1.0``). If ``False``, use ``1.0`` everywhere (classic PPI).
        metric_lower_bound : float, optional
            Known lower bound of the metric. Defaults to ``0.0``.
        metric_upper_bound : float, optional
            Known upper bound of the metric. Defaults to ``1.0``.

        Returns
        -------
        PredictionPoweredMeanMonitoringResult
            Per-batch estimates, running means, anytime-valid confidence bounds,
            alarm flags, and the alarm threshold, all in the original metric
            orientation.

        Raises
        ------
        ValueError
            - If ``y_true`` is empty.
            - If ``y_true``, ``y_proxy`` and ``batches`` have different lengths.
            - If ``batches`` contains NaN values (numeric dtype) or None values (non-numeric dtype).
            - If ``confidence_level`` is not in (0, 1).
            - If ``metric_lower_bound >= metric_upper_bound``.
            - If ``threshold`` falls outside ``[metric_lower_bound, metric_upper_bound]``.
            - If any proxy value is NaN or all proxy values are identical.
            - If labeled ``y_true`` values are constant.
            - If any value falls outside ``[metric_lower_bound, metric_upper_bound]``.
            - If batches are interleaved rather than grouped into contiguous blocks.
            - If any batch has fewer than 2 labeled or fewer than 2 unlabeled samples.
            - If proxy values are constant across the prior batches (with ``power_tuning=True``).
        """
        risk_y_true, risk_y_proxy, risk_threshold, batch_codes, batch_n_true, batch_n_proxy = _preprocess(
            y_true,
            y_proxy,
            batches,
            higher_is_better,
            threshold,
            confidence_level,
            metric_lower_bound,
            metric_upper_bound,
        )
        risk_batch_estimates, _ = _compute_batch_estimates(risk_y_true, risk_y_proxy, batch_codes, power_tuning)
        clipped_risk_batch_estimates = np.clip(risk_batch_estimates, 0.0, 1.0)

        miscoverage = 1.0 - confidence_level
        risk_running_means, risk_lower_bounds = _compute_empirical_bernstein_bounds(
            clipped_risk_batch_estimates, risk_threshold, miscoverage
        )
        running_means, confidence_bounds, batch_mean_estimates = _postprocess(
            risk_running_means,
            risk_lower_bounds,
            risk_batch_estimates,
            higher_is_better,
            metric_lower_bound,
            metric_upper_bound,
        )
        confidence_sequence = EmpiricalBernsteinConfidenceSequence(
            running_mean_estimates=running_means,
            confidence_bounds=confidence_bounds,
        )
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
