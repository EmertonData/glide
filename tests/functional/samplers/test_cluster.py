import numpy as np

from glide.samplers import UniformClusterSampler, UniformSampler


def test_sample_preserves_cluster_integrity():
    clusters = np.array(["A", "A", "B", "B", "B", "C", "C", "C", "C"], dtype=object)
    sampler = UniformClusterSampler()
    xi = sampler.sample(clusters, n_clusters=2, random_seed=0)

    for cluster_id in np.unique(clusters):
        cluster_xi_values = xi[clusters == cluster_id]
        assert len(np.unique(cluster_xi_values)) == 1


def test_sample_unit_clusters_matches_uniform_sampler_distribution():
    clusters = np.array(["A", "B", "C", "D", "E", "F"], dtype=object)
    n_clusters = 3
    n_total = len(clusters)
    n_seeds = 5_000

    cluster_counts = np.zeros(n_total, dtype=int)
    uniform_counts = np.zeros(n_total, dtype=int)
    for seed in range(n_seeds):
        cluster_counts += UniformClusterSampler().sample(clusters, n_clusters=n_clusters, random_seed=seed)
        uniform_counts += UniformSampler().sample(n_samples=n_total, budget=n_clusters, random_seed=seed).astype(int)

    expected_frequency = n_clusters / n_total
    np.testing.assert_allclose(cluster_counts / n_seeds, expected_frequency, atol=0.02)
    np.testing.assert_allclose(uniform_counts / n_seeds, expected_frequency, atol=0.02)
