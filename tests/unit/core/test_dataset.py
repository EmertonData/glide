import numpy as np
import pytest

from glide.core.dataset import Dataset


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


def test_getitem_int_out_of_range_returns_empty_dataset():
    result = Dataset(RECORDS)[10]
    assert result == Dataset()


def test_getitem_unknown_field_returns_nan():
    result = Dataset(RECORDS)["unknown"]
    np.testing.assert_array_equal(result, np.array([np.nan, np.nan]))


def test_getitem_list_missing_only_returns_nan_matrix():
    result = Dataset(RECORDS)[["unknown"]]
    expected = np.array([[np.nan], [np.nan]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_getitem_list_of_strings_returns_2d_array_with_nan_for_missing():
    result = Dataset(RECORDS)[["human", "llm"]]
    expected = np.array([[0, 0], [np.nan, 1]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_getitem_list_of_strings_returns_2d_array_with_nan_for_missing_column():
    result = Dataset(RECORDS)[["human", "llm", "unknown"]]
    expected = np.array([[0, 0, np.nan], [np.nan, 1, np.nan]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_getitem_boolean_array_returns_filtered_dataset():
    result = Dataset(RECORDS)[Dataset(RECORDS)["human"] == 0]
    assert result == Dataset([{"human": 0, "llm": 0}])


def test_getitem_unsupported_key_type_raises():
    with pytest.raises(TypeError):
        Dataset(RECORDS)[3.14]  # type: ignore[index]


def test_to_numpy_human_then_llm():
    result = Dataset(RECORDS).to_numpy(fields=["human", "llm"])
    expected = np.array([[0, 0], [np.nan, 1]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_to_numpy_llm_then_human():
    result = Dataset(RECORDS).to_numpy(fields=["llm", "human"])
    expected = np.array([[0, 0], [1, np.nan]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_to_numpy_single_field():
    result = Dataset(RECORDS).to_numpy(fields=["llm"])
    expected = np.array([[0], [1]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_to_numpy_unknown_field_raises():
    with pytest.raises(ValueError):
        Dataset(RECORDS).to_numpy(fields=["unknown"])
