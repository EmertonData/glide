import numpy as np
import pytest

from glide.estimators.stratified_core import preprocess

# ── helpers ────────────────────────────────────────────────────────────────────


@pytest.fixture
def y_true() -> np.ndarray:
    return np.array([5.0, 6.0, np.nan, np.nan, 5.0, 6.0, np.nan, np.nan])


@pytest.fixture
def y_proxy() -> np.ndarray:
    return np.array([4.9, 6.1, 5.2, 6.1, 4.9, 6.1, 5.2, 6.1])


@pytest.fixture
def groups() -> np.ndarray:
    return np.array(["A", "A", "A", "A", "B", "B", "B", "B"])


# --- preprocess ---


def test_preprocess_returns_correct_shapes(y_true, y_proxy, groups):
    strata = preprocess(y_true, y_proxy, groups)
    assert len(strata) == 2
    for y_true_labeled, y_proxy_labeled, y_proxy_unlabeled in strata:
        assert len(y_true_labeled) == 2
        assert len(y_proxy_labeled) == 2
        assert len(y_proxy_unlabeled) == 2


def test_preprocess_raises_on_length_mismatch():
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([1.1, 1.8, 3.9])
    grps = np.array(["A", "A", "A", "A"])
    with pytest.raises(ValueError, match="y_true, y_proxy, and groups must have the same length"):
        preprocess(y_true, y_proxy, grps)


def test_preprocess_raises_on_nan_proxy():
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([1.1, np.nan, 5.2, 6.1])
    grps = np.array(["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="Input proxy values contain NaN"):
        preprocess(y_true, y_proxy, grps)


def test_preprocess_raises_on_zero_variance_proxy_in_stratum():
    y_true = np.array([1.0, 2.0, np.nan, np.nan])
    y_proxy = np.array([1.0, 1.0, 1.0, 1.0])
    grps = np.array(["A", "A", "A", "A"])
    with pytest.raises(ValueError, match="Input proxy values have zero variance in stratum 'A'"):
        preprocess(y_true, y_proxy, grps)


def test_preprocess_raises_on_stratum_size_too_small():
    y_true = np.array([1.0, np.nan, 4.0, np.nan])
    y_proxy = np.array([1.1, 1.8, 3.9, 4.8])
    grps = np.array(["A", "A", "B", "B"])
    with pytest.raises(ValueError, match="Too few labeled or unlabeled samples in stratum 'A'"):
        preprocess(y_true, y_proxy, grps)
