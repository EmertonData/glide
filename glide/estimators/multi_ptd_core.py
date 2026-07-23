import numpy as np
from numpy.typing import NDArray


def _compute_tuning_parameters(
    bootstrap_y_true_means: NDArray,
    bootstrap_y_proxies_labeled_means: NDArray,
    cov_matrix_proxies_unlabeled: NDArray,
    power_tuning: bool,
) -> NDArray:
    n_proxies = bootstrap_y_proxies_labeled_means.shape[1]
    if not power_tuning:
        lambdas_ = np.full(n_proxies, 1 / np.sqrt(n_proxies))
        return lambdas_
    combined = np.vstack([bootstrap_y_true_means, bootstrap_y_proxies_labeled_means.T])
    cov_all = np.cov(combined, ddof=1)
    cov_vec = cov_all[0, 1:]
    cov_matrix_labeled = cov_all[1:, 1:]
    denom_matrix = np.atleast_2d(cov_matrix_labeled) + np.atleast_2d(cov_matrix_proxies_unlabeled)
    try:
        lambdas_ = np.linalg.solve(denom_matrix, cov_vec)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "The combined proxy covariance matrix is singular. "
            "This typically means two or more proxy columns are perfectly correlated."
        ) from exc
    return lambdas_


def _compute_bootstrap_mean_estimates(
    bootstrap_y_true_means: NDArray,
    bootstrap_y_proxies_labeled_means: NDArray,
    mean_proxies_unlabeled: NDArray,
    cov_matrix_proxies_unlabeled: NDArray,
    lambdas_: NDArray,
    rng: np.random.Generator,
) -> NDArray:
    n_bootstrap = len(bootstrap_y_true_means)
    z = rng.standard_normal(n_bootstrap)
    var_unlabeled = lambdas_ @ cov_matrix_proxies_unlabeled @ lambdas_
    unlabeled_means = lambdas_ @ mean_proxies_unlabeled + z * np.sqrt(var_unlabeled)
    rectifiers = bootstrap_y_true_means - bootstrap_y_proxies_labeled_means @ lambdas_
    bootstrap_mean_estimates = unlabeled_means + rectifiers
    return bootstrap_mean_estimates
