from unittest.mock import patch

import numpy as np
import pytest

import glide.simulators.stratified_multi_binary as stratified_multi_binary_module
from glide.simulators import generate_stratified_multi_binary_dataset


@pytest.fixture
def dataset_params():
    return [3, 3], [0.6, 0.8], [[0.5, 0.55], [0.7, 0.75]], [[0.75, 0.7], [0.75, 0.7]]


def test_generate_stratified_multi_binary_dataset_structure_and_counts(dataset_params):
    y_true, y_proxies, groups = generate_stratified_multi_binary_dataset(*dataset_params, random_seed=0)
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_proxies, np.ndarray)
    assert isinstance(groups, np.ndarray)
    assert y_true.shape == (6,)
    assert y_proxies.shape == (6, 2)
    assert groups.shape == (6,)
    y_true_expected = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    y_proxies_expected = [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    np.testing.assert_array_equal(groups, [0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(y_true, y_true_expected)
    np.testing.assert_allclose(y_proxies, y_proxies_expected)


def test_generate_stratified_multi_binary_dataset_delegates_validation(dataset_params):
    n_samples, true_mean, proxy_means, correlations = dataset_params

    with (
        patch.object(stratified_multi_binary_module, "_validate_non_empty") as mock_validate_non_empty,
        patch.object(stratified_multi_binary_module, "_validate_is_2d") as mock_validate_is_2d,
        patch.object(stratified_multi_binary_module, "_validate_equal_lengths") as mock_validate_equal_lengths,
    ):
        generate_stratified_multi_binary_dataset(*dataset_params)

        mock_validate_non_empty.assert_called_once()
        np.testing.assert_array_equal(mock_validate_non_empty.call_args[0][0], np.array(n_samples, dtype=int))
        assert mock_validate_non_empty.call_args[0][1] == "n_samples"

        assert mock_validate_is_2d.call_count == 2
        np.testing.assert_array_equal(mock_validate_is_2d.call_args_list[0].args[0], np.array(proxy_means))
        assert mock_validate_is_2d.call_args_list[0].args[1] == "proxy_means"
        np.testing.assert_array_equal(mock_validate_is_2d.call_args_list[1].args[0], np.array(correlations))
        assert mock_validate_is_2d.call_args_list[1].args[1] == "correlations"

        mock_validate_equal_lengths.assert_called_once()
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][0], np.array(n_samples))
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][1], np.array(true_mean))
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][2], np.array(proxy_means))
        np.testing.assert_array_equal(mock_validate_equal_lengths.call_args[0][3], np.array(correlations))
        expected_names = ["n_samples", "true_mean", "proxy_means", "correlations"]
        assert mock_validate_equal_lengths.call_args[1]["names"] == expected_names


def test_generate_stratified_multi_binary_dataset_reproducibility(dataset_params):
    y_true1, y_proxies1, groups1 = generate_stratified_multi_binary_dataset(*dataset_params, random_seed=42)
    y_true2, y_proxies2, groups2 = generate_stratified_multi_binary_dataset(*dataset_params, random_seed=42)
    np.testing.assert_allclose(y_true1, y_true2)
    np.testing.assert_allclose(y_proxies1, y_proxies2)
    np.testing.assert_array_equal(groups1, groups2)


def test_generate_stratified_multi_binary_dataset_different_seed_results_differ(dataset_params):
    y_true1, y_proxies1, _ = generate_stratified_multi_binary_dataset(*dataset_params, random_seed=0)
    y_true2, y_proxies2, _ = generate_stratified_multi_binary_dataset(*dataset_params, random_seed=1)
    assert not np.array_equal(y_true1, y_true2) or not np.array_equal(y_proxies1, y_proxies2)
