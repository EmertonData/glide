import numpy as np
from numpy.typing import NDArray


def _compute_tuning_parameters(
    y_true: NDArray,
    y_proxies_labeled: NDArray,
    y_proxies_unlabeled: NDArray,
    power_tuning: bool,
) -> NDArray:
    n_labeled = len(y_true)
    n_unlabeled = len(y_proxies_unlabeled)
    M = y_proxies_labeled.shape[1]
    if not power_tuning:
        lambdas_ = np.full(M, 1 / np.sqrt(M))
        return lambdas_
    y_proxies_all = np.vstack([y_proxies_labeled, y_proxies_unlabeled])
    proxy_cov_matrix = np.atleast_2d(np.cov(y_proxies_all.T, ddof=1))
    centered_proxies_labeled = y_proxies_labeled - np.mean(y_proxies_labeled, axis=0)
    centered_true = y_true - np.mean(y_true)
    proxy_true_cov = centered_proxies_labeled.T @ centered_true / (n_labeled - 1)
    try:
        lambdas_ = n_unlabeled / (n_labeled + n_unlabeled) * np.linalg.solve(proxy_cov_matrix, proxy_true_cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The proxy covariance matrix is singular. "
            "This typically means two or more proxy columns are perfectly correlated."
        ) from exc
    return lambdas_


def _compute_mean_estimate(
    y_true: NDArray,
    y_proxies_labeled: NDArray,
    y_proxies_unlabeled: NDArray,
    lambdas_: NDArray,
) -> float:
    rectifier = np.mean(y_true) - lambdas_ @ np.mean(y_proxies_labeled, axis=0)
    proxy_mean = lambdas_ @ np.mean(y_proxies_unlabeled, axis=0)
    mean_estimate = proxy_mean + rectifier
    return mean_estimate


def _compute_std_estimate(
    y_true: NDArray,
    y_proxies_labeled: NDArray,
    y_proxies_unlabeled: NDArray,
    lambdas_: NDArray,
) -> float:
    n_labeled = len(y_true)
    n_unlabeled = len(y_proxies_unlabeled)
    rectifier_residuals = y_true - y_proxies_labeled @ lambdas_
    rectifier_var = np.var(rectifier_residuals, ddof=1) / n_labeled
    proxy_projections = y_proxies_unlabeled @ lambdas_
    proxy_var = np.var(proxy_projections, ddof=1) / n_unlabeled
    var_estimate = rectifier_var + proxy_var
    std_estimate = np.sqrt(var_estimate)
    return std_estimate
