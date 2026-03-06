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
