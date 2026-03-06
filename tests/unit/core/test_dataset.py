from glide.core.dataset import Dataset


def test_dataset_constructor():
    dataset = Dataset()
    dataset += [{"a": 0}]
    dataset *= 2
    assert dataset == [{"a": 0}, {"a": 0}]
