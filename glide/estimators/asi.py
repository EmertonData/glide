from math import floor

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import CLTConfidenceInterval
from glide.engines.asi import ASIMeanEngine
from glide.engines.ipw_classical import IPWClassicalMeanEngine
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult


class ASIMeanEstimator:
    """Estimator for population mean using Active Statistical Inference (ASI).

    This class implements the ASI method which extends PPI++ to non-uniform sampling.
    Each labeled sample has a known, pre-determined sampling probability π_i. Inverse
    probability weighting (IPW) corrects for this non-uniform selection, yielding valid
    confidence intervals under any sampling rule.

    The special case where all π_i are equal to n_labeled / n recovers PPI++ at λ = 1.

    References
    ----------
    Zrnic, Tijana, and Emmanuel J. Candès. "Active statistical inference." In Proceedings
    of the 41st International Conference on Machine Learning, pp. 62993-63010. 2024.

    Gligorić, Kristina, Tijana Zrnic, Cinoo Lee, Emmanuel Candes, and Dan Jurafsky.
    "Can unconfident llm annotations be used for confident conclusions?." In Proceedings
    of the 2025 Conference of the Nations of the Americas Chapter of the Association for
    Computational Linguistics: Human Language Technologies (Volume 1: Long Papers),
    pp. 3514-3533. 2025.

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators import ASIMeanEstimator
    >>> y_true = np.array([0.0, 1.0, np.nan, np.nan])
    >>> y_proxy = np.array([0.1, 0.9, 0.5, 0.5])
    >>> sampling_probability = np.array([0.8, 0.8, 0.8, 0.8])
    >>> estimator = ASIMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxy, sampling_probability)
    >>> print(result)
    Metric: Metric
    Point Estimate: 0.548
    Confidence Interval (95%): [0.138, 0.958]
    Estimator : ASIMeanEstimator
    n_true: 2
    n_proxy: 4
    Effective Sample Size: 4
    """

    def __init__(self) -> None:
        self._engine = ASIMeanEngine()
        self._classical_engine = IPWClassicalMeanEngine()

    def estimate(
        self,
        y_true: NDArray,
        y_proxy: NDArray,
        sampling_probability: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        *,
        power_tuning: bool = True,
    ) -> PredictionPoweredMeanInferenceResult:
        """Estimate the population mean using Active Statistical Inference (ASI).

        Uses inverse-probability weighting (IPW) to correct for non-uniform sampling,
        combining labeled and unlabeled samples into a single IPW-corrected estimator.
        A power-tuning step (enabled by default) finds the λ that minimises asymptotic
        variance.

        Parameters
        ----------
        y_true : NDArray
            Array of shape ``(n_samples,)`` with ground-truth labels. Use ``np.nan`` for
            unlabeled samples (ξ_i = 0); non-NaN entries are treated as labeled (ξ_i = 1).
        y_proxy : NDArray
            Array of shape ``(n_samples,)`` with proxy predictions. Must be present for every
            sample and must not contain NaN.
        sampling_probability : NDArray
            Array of shape ``(n_samples,)`` with the pre-determined sampling probability
            π_i ∈ [0, 1] for each sample. Entries with π_i = 0 are excluded from all
            computations.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        power_tuning : bool, optional
            If ``True`` (default), selects λ analytically to minimise asymptotic variance.
            If ``False``, uses λ = 1 (plain IPW estimator).

        Returns
        -------
        PredictionPoweredMeanInferenceResult
            Contains a ``CLTConfidenceInterval``, metric name, estimator
            name (``"ASIMeanEstimator"``), and counts ``n_true`` (labeled samples) and
            ``n_proxy`` (total samples).

        Raises
        ------
        ValueError
            - If ``y_true``, ``y_proxy``, and ``sampling_probability`` do not all have the same length.
            - If any proxy value is NaN.
            - If fewer than two labeled or two unlabeled samples remain after excluding zero-probability samples.
            - If ``power_tuning`` is ``True`` and the rectifiers ``y_proxy * (ξ_i / π_i - 1)`` are constant.
            - If any value in ``sampling_probability`` is not in [0, 1].
        """
        asi_dataset = self._engine._preprocess(y_true, y_proxy, sampling_probability)
        y_true_filtered, _, sampling_probability_filtered = asi_dataset
        n_labeled = int(np.count_nonzero(~np.isnan(y_true_filtered)))
        n_proxy = len(sampling_probability_filtered)

        tuning_parameter = self._engine._fit_tuning_parameter(asi_dataset, power_tuning=power_tuning)
        mean, std = self._engine._compute_mean_and_std(asi_dataset, tuning_parameter)
        confidence_interval = CLTConfidenceInterval(mean=mean, std=std, confidence_level=confidence_level)

        ipw_classical_dataset = (y_true_filtered, sampling_probability_filtered)
        _, classical_std = self._classical_engine._compute_mean_and_std(ipw_classical_dataset, None)
        effective_sample_size = floor(n_labeled * classical_std**2 / std**2)

        return PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_labeled,
            n_proxy=n_proxy,
            effective_sample_size=effective_sample_size,
        )
