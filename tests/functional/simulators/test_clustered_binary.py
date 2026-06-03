import numpy as np
import pytest

from glide.simulators import generate_clustered_binary_dataset, simulate_annotation


def test_generate_clustered_binary_dataset_empirical_means_and_correlation():
    y_true_oracle, y_proxy, clusters = generate_clustered_binary_dataset(
        n_total=5000, n_clusters=50, true_mean=0.7, proxy_mean=0.6, correlation=0.8, random_seed=42
    )
    assert np.mean(y_true_oracle) == pytest.approx(0.7, abs=0.03)
    assert np.mean(y_proxy) == pytest.approx(0.6, abs=0.03)
    assert np.corrcoef(y_true_oracle, y_proxy)[0, 1] == pytest.approx(0.8, abs=0.05)


def test_generate_clustered_binary_dataset_no_partial_clusters_after_annotation():
    y_true_oracle, y_proxy, clusters = generate_clustered_binary_dataset(n_total=100, n_clusters=10, random_seed=0)
    unique_clusters = np.unique(clusters)
    rng = np.random.default_rng(1)
    labeled_clusters = rng.choice(unique_clusters, size=len(unique_clusters) // 2, replace=False)
    xi = np.where(np.isin(clusters, labeled_clusters), 1.0, 0.0)

    y_true = simulate_annotation(y_true_oracle, xi)

    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        cluster_labels = y_true[cluster_mask]
        is_labeled = ~np.isnan(cluster_labels)
        assert is_labeled.all() or (~is_labeled).all()
