from typing import Dict, List, SupportsIndex, overload

import numpy as np
from numpy.typing import NDArray


class Dataset(list):
    @property
    def records(self) -> List[Dict]:
        return list(self)

    @overload
    def __getitem__(self, key: SupportsIndex) -> Dict: ...
    @overload
    def __getitem__(self, key: slice) -> List[Dict]: ...
    @overload
    def __getitem__(self, key: str) -> NDArray: ...
    def __getitem__(self, key):
        """Access records or a column from the dataset.

        Parameters
        ----------
        key : int or slice or str
            - ``int``: returns the record at that position as a ``Dict``.
            - ``slice``: returns a list of records as ``List[Dict]``.
            - ``str``: returns all values for that field as a 1D ``NDArray``.

        Returns
        -------
        Dict or List[Dict] or NDArray
            See ``key`` above.

        Examples
        --------
        >>> from glide.core.dataset import Dataset
        >>> dataset = Dataset([{"score": i} for i in range(5)])
        >>> dataset[0]           # first record → Dict
        {'score': 0}
        >>> dataset[0:3]         # first five records → List[Dict]
        [{'score': 0}, {'score': 1}, {'score': 2}]
        >>> dataset["score"]     # "score" column → NDArray
        array([0., 1., 2., 3., 4.])
        """

        # If key is a string return a column, else if an integer index return a record
        if isinstance(key, str):
            return self.to_numpy([key])[:, 0]
        return super().__getitem__(key)

    def to_numpy(self, fields: List[str]) -> NDArray:
        """Convert the dataset to a 2D numpy array of floats.

        Parameters
        ----------
        fields : List[str]
            Ordered list of record keys to use as columns. Missing values are filled with NaN.

        Returns
        -------
        NDArray
            2D float array of shape (n_records, n_fields).

        Raises
        ------
        ValueError
            If a field is not present in any record.
        """
        rows = []
        for record in self:
            row = [record.get(field, np.nan) for field in fields]
            rows.append(row)
        result = np.array(rows, dtype=float)

        nan_counts = np.isnan(result).sum(axis=0)
        all_nan_mask = nan_counts == len(result)
        unknown_fields = np.array(fields)[all_nan_mask].tolist()
        if unknown_fields:
            raise ValueError(f"Unknown fields: {', '.join(unknown_fields)}")

        return result
