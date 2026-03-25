import numpy as np
import pytest

from glide.core.dataset import Dataset


@pytest.fixture
def records():
    return [
        {"human": 0, "llm": 0},
        {"llm": 1},
    ]


def test_dataset_empty():
    dataset = Dataset()
    assert dataset == []


def test_dataset_iadd():
    dataset = Dataset()
    dataset += [{"a": 0}]
    assert dataset == [{"a": 0}]


def test_dataset_imul():
    dataset = Dataset([{"a": 0}])
    dataset *= 2
    assert dataset == [{"a": 0}, {"a": 0}]


def test_dataset_records():
    dataset = Dataset([{"a": 0}, {"b": 1}])
    assert dataset.records == [{"a": 0}, {"b": 1}]


def test_dataset_add():
    dataset1 = Dataset([{"a": 0}])
    dataset2 = Dataset([{"b": 1}])
    result = dataset1 + dataset2
    assert result == [{"a": 0}, {"b": 1}]
    assert isinstance(result, Dataset)


def test_dataset_radd():
    dataset = Dataset([{"b": 1}])
    result = [{"a": 0}] + dataset
    assert result == [{"a": 0}, {"b": 1}]
    assert isinstance(result, Dataset)


RECORDS = [
    {"human": 0, "llm": 0},
    {"llm": 1},
]


def test_getitem_string_returns_column_with_nan_for_missing():
    result = Dataset(RECORDS)["human"]
    np.testing.assert_array_equal(result, np.array([0.0, np.nan]))


def test_getitem_int_returns_record():
    result = Dataset(RECORDS)[1]
    assert result == Dataset([{"llm": 1}])


def test_getitem_slice_returns_records():
    result = Dataset(RECORDS)[0:2]
    assert result == Dataset([{"human": 0, "llm": 0}, {"llm": 1}])


def test_getitem_list_of_strings_returns_2d_array():
    result = Dataset(RECORDS)[["human", "llm"]]
    expected = np.array([[0, 0], [np.nan, 1]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_to_numpy_llm_then_human(records):
    result = Dataset(records).to_numpy(fields=["llm", "human"])
    expected = np.array([[0, 0], [1, np.nan]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_to_numpy_single_field(records):
    result = Dataset(records).to_numpy(fields=["llm"])
    expected = np.array([[0], [1]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_to_numpy_unknown_field_raises(records):
    with pytest.raises(ValueError):
        Dataset(records).to_numpy(fields=["unknown"])
