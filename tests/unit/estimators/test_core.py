import numpy as np

from glide.estimators.core import _split_labeled_unlabeled


def test_split_labeled_unlabeled():
    y_true = np.array([1.0, np.nan, 2.0, np.nan])
    y_proxy = np.array([0.9, 1.1, 2.1, 3.0])

    y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, labeled_mask = _split_labeled_unlabeled(y_true, y_proxy)

    np.testing.assert_array_equal(y_true_labeled, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(y_proxy_labeled, np.array([0.9, 2.1]))
    np.testing.assert_array_equal(y_proxy_unlabeled, np.array([1.1, 3.0]))
    np.testing.assert_array_equal(labeled_mask, np.array([True, False, True, False]))


def test_split_labeled_unlabeled_2d_y_proxy():
    y_true = np.array([1.0, np.nan, 2.0, np.nan])
    y_proxy = np.array([[0.9, 0.8], [1.1, 1.2], [2.1, 2.2], [3.0, 3.1]])

    y_true_labeled, y_proxy_labeled, y_proxy_unlabeled, labeled_mask = _split_labeled_unlabeled(y_true, y_proxy)

    np.testing.assert_array_equal(y_true_labeled, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(y_proxy_labeled, np.array([[0.9, 0.8], [2.1, 2.2]]))
    np.testing.assert_array_equal(y_proxy_unlabeled, np.array([[1.1, 1.2], [3.0, 3.1]]))
    np.testing.assert_array_equal(labeled_mask, np.array([True, False, True, False]))
