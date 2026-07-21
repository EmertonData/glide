from math import floor
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from glide.confidence_intervals import BootstrapConfidenceInterval
from glide.core.validation import (
    _validate_equal_lengths,
    _validate_sample_sizes,
    _validate_y_proxies,
    _validate_y_true,
)
from glide.estimators.classical import ClassicalMeanEstimator
from glide.estimators.core import _split_labeled_unlabeled
from glide.estimators.multi_ptd_core import (
    _compute_bootstrap_mean_estimates,
    _compute_tuning_parameters,
)
from glide.estimators.ptd_core import _compute_bootstrap_labeled_means
from glide.mean_inference_results import PredictionPoweredMeanInferenceResult


class MultiPTDMeanEstimator:
    """Estimator for population mean using Predict-Then-Debias with multiple proxies.

    Extends PTDMeanEstimator to settings where M >= 1 proxy predictors are
    available. The optimal weight vector lambda is estimated from the bootstrap
    and used to form a single combined proxy before applying the PTD rectifier.
    Confidence intervals are constructed via a bootstrap percentile method.

    The bootstrap uses a CLT-based algorithm for the unlabeled contribution:
    the combined unlabeled proxy mean (lambda times the mean of y_proxies_unlabeled)
    is drawn from a scalar Gaussian at each iteration, keeping the per-iteration cost
    O(n_labeled), where n_labeled and n_unlabeled are the number of labeled
    and unlabeled samples respectively.

    When M = 1, the estimator is equivalent to PTDMeanEstimator with the same
    power_tuning setting and the same random seed.

    References
    ----------
    Kluger, Dan M., Kerri Lu, Tijana Zrnic, Sherrie Wang, and Stephen Bates.
    "Prediction-powered inference with imputed covariates and nonuniform sampling."
    arXiv preprint arXiv:2501.18577 (2025).

    Shan, Jiawei, Zhifeng Chen, Yiming Dong, Yazhen Wang, and Jiwei Zhao.
    "SADA: Safe and Adaptive Aggregation of Multiple Black-Box Predictions in Semi-Supervised Learning."
    arXiv preprint arXiv:2509.21707 (2025).

    Examples
    --------
    >>> import numpy as np
    >>> from glide.estimators import MultiPTDMeanEstimator
    >>> y_true = np.array([5.0, 6.0, np.nan, np.nan])
    >>> y_proxies = np.array([[4.9], [6.1], [5.2], [6.1]])
    >>> estimator = MultiPTDMeanEstimator()
    >>> result = estimator.estimate(y_true, y_proxies, n_bootstrap=5, random_seed=0)
    >>> print(result.estimator_name)
    MultiPTDMeanEstimator
    """

    def _preprocess(
        self,
        y_true_all: NDArray,
        y_proxies_all: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        _validate_equal_lengths(y_true_all, y_proxies_all, names=["y_true", "y_proxies"])
        _validate_y_proxies(y_proxies_all)
        _validate_y_true(y_true_all)
        y_true, y_proxies_labeled, y_proxies_unlabeled, labeled_mask = _split_labeled_unlabeled(
            y_true_all, y_proxies_all
        )
        _validate_sample_sizes(labeled_mask)
        return y_true, y_proxies_labeled, y_proxies_unlabeled

    def estimate(
        self,
        y_true: NDArray,
        y_proxies: NDArray,
        metric_name: str = "Metric",
        confidence_level: float = 0.95,
        n_bootstrap: int = 2000,
        power_tuning: bool = True,
        random_seed: Optional[int] = None,
    ) -> PredictionPoweredMeanInferenceResult:
        """Estimate the population mean using Multi-PTD.

        Combines a small set of labeled samples with a large set of unlabeled
        samples, leveraging M proxy predictors simultaneously. The optimal
        weight vector lambda and the confidence interval are both derived from a
        bootstrap over the labeled set. The sampling variability of each
        unlabeled proxy mean is approximated via a single scalar Gaussian draw
        per iteration, keeping the per-iteration cost O(n_labeled).

        Parameters
        ----------
        y_true : NDArray
            Array of observations, shape ``(n_samples,)``.
            Labeled entries are finite; unlabeled entries are ``np.nan``.
        y_proxies : NDArray
            2D array of proxy predictions, shape ``(n_samples, M)``.
            Must be fully populated (no NaN). Each column must have nonzero variance.
        metric_name : str, optional
            Human-readable label for the metric. Defaults to ``"Metric"``.
        confidence_level : float, optional
            Target coverage for the confidence interval. Defaults to ``0.95``.
        n_bootstrap : int, optional
            Number of bootstrap resamples. Defaults to ``2000``.
        power_tuning : bool, optional
            If ``True`` (default), estimate the optimal lambda from the bootstrap
            covariances. If ``False``, set all entries of lambda to ``1/sqrt(M)``.
        random_seed : int, optional
            Seed for the random number generator. Defaults to ``None``.

        Returns
        -------
        PredictionPoweredMeanInferenceResult
            Contains a ``BootstrapConfidenceInterval``, metric name, estimator
            name (``"MultiPTDMeanEstimator"``), and counts ``n_true`` (labeled
            samples) and ``n_proxy`` (total samples).

        Raises
        ------
        ValueError
            - If ``y_true`` and ``y_proxies`` have different lengths.
            - If ``y_proxies`` is not a 2D array.
            - If any value in ``y_proxies`` is NaN.
            - If any column of ``y_proxies`` is constant.
            - If ``y_true`` contains only NaN or its labeled values are constant.
            - If there are fewer than 2 labeled or fewer than 2 unlabeled samples.
            - If the combined proxy covariance matrix is singular.
        """
        y_true_filtered, y_proxies_labeled, y_proxies_unlabeled = self._preprocess(y_true, y_proxies)
        n_labeled, n_unlabeled = len(y_true_filtered), len(y_proxies_unlabeled)
        rng = np.random.default_rng(random_seed)

        mean_proxies_unlabeled = np.mean(y_proxies_unlabeled, axis=0)
        cov_matrix_proxies_unlabeled = np.atleast_2d(np.cov(y_proxies_unlabeled, rowvar=False, ddof=1)) / n_unlabeled

        bootstrap_y_true_means, bootstrap_y_proxies_labeled_means = _compute_bootstrap_labeled_means(
            y_true_filtered, y_proxies_labeled, n_bootstrap, rng
        )
        lambdas_ = _compute_tuning_parameters(
            bootstrap_y_true_means, bootstrap_y_proxies_labeled_means, cov_matrix_proxies_unlabeled, power_tuning
        )
        bootstrap_mean_estimates = _compute_bootstrap_mean_estimates(
            bootstrap_y_true_means,
            bootstrap_y_proxies_labeled_means,
            mean_proxies_unlabeled,
            cov_matrix_proxies_unlabeled,
            lambdas_,
            rng,
        )

        confidence_interval = BootstrapConfidenceInterval(
            bootstrap_estimates=bootstrap_mean_estimates,
            confidence_level=confidence_level,
        )
        classical_confidence_interval = ClassicalMeanEstimator().estimate(y_true_filtered).confidence_interval
        effective_sample_size = floor(n_labeled * classical_confidence_interval.var / confidence_interval.var)
        result = PredictionPoweredMeanInferenceResult(
            confidence_interval=confidence_interval,
            metric_name=metric_name,
            estimator_name=self.__class__.__name__,
            n_true=n_labeled,
            n_proxy=n_labeled + n_unlabeled,
            effective_sample_size=effective_sample_size,
        )
        return result
