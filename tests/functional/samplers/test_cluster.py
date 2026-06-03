import numpy as np
import pytest

from glide.samplers import ClusterSampler, UniformSampler


@pytest.mark.parametrize("strategy", ["uniform", "proportional"])
def test_sample_preserves_cluster_integrity(strategy):
    clusters = np.array(["A", "A", "B", "B", "B", "C", "C", "C", "C"], dtype=object)
    sampler = ClusterSampler()
    xi = sampler.sample(clusters, n_clusters=2, strategy=strategy, random_seed=0)

    for cluster_id in np.unique(clusters):
        cluster_xi_values = xi[clusters == cluster_id]
        assert len(np.unique(cluster_xi_values)) == 1


def test_sample_proportional_equals_uniform_for_equal_size_clusters():
    clusters = np.array(["A", "A", "B", "B", "C", "C"], dtype=object)
    n_clusters = 2
    n_total_clusters = 3
    n_seeds = 5_000

    uniform_counts = np.zeros(n_total_clusters, dtype=int)
    proportional_counts = np.zeros(n_total_clusters, dtype=int)
    for seed in range(n_seeds):
        xi_uniform = ClusterSampler().sample(clusters, n_clusters=n_clusters, strategy="uniform", random_seed=seed)
        xi_proportional = ClusterSampler().sample(
            clusters, n_clusters=n_clusters, strategy="proportional", random_seed=seed
        )
        for i, cluster_id in enumerate(["A", "B", "C"]):
            uniform_counts[i] += xi_uniform[clusters == cluster_id][0]
            proportional_counts[i] += xi_proportional[clusters == cluster_id][0]

    expected_frequency = n_clusters / n_total_clusters
    np.testing.assert_allclose(uniform_counts / n_seeds, expected_frequency, atol=0.02)
    np.testing.assert_allclose(proportional_counts / n_seeds, expected_frequency, atol=0.02)


def test_sample_unit_clusters_matches_uniform_sampler_distribution():
    clusters = np.array(["A", "B", "C", "D", "E", "F"], dtype=object)
    n_clusters = 3
    n_total = len(clusters)
    n_seeds = 5_000

    cluster_counts = np.zeros(n_total, dtype=int)
    uniform_counts = np.zeros(n_total, dtype=int)
    for seed in range(n_seeds):
        cluster_counts += ClusterSampler().sample(clusters, n_clusters=n_clusters, strategy="uniform", random_seed=seed)
        uniform_counts += UniformSampler().sample(n_samples=n_total, budget=n_clusters, random_seed=seed).astype(int)

    expected_frequency = n_clusters / n_total
    np.testing.assert_allclose(cluster_counts / n_seeds, expected_frequency, atol=0.02)
    np.testing.assert_allclose(uniform_counts / n_seeds, expected_frequency, atol=0.02)
