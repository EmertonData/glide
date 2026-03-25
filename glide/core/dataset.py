from typing import Dict, List, SupportsIndex, overload

import numpy as np
from numpy.typing import NDArray


class Dataset(list):
    @property
    def records(self) -> List[Dict]:
        return list(self)

    @overload
    def __getitem__(self, key: SupportsIndex) -> "Dataset": ...
    @overload
    def __getitem__(self, key: slice) -> "Dataset": ...
    @overload
    def __getitem__(self, key: NDArray[np.bool_]) -> "Dataset": ...
    @overload
    def __getitem__(self, key: str) -> NDArray: ...
    @overload
    def __getitem__(self, key: List[str]) -> NDArray: ...
    def __getitem__(self, key):
        """Access records or a column from the dataset.

        Parameters
        ----------
        key : int or slice or str or list[str]
            - int: returns a Dataset containing a single record.
                   out-of-range int returns an empty Dataset.
            - slice: returns a Dataset subset over the requested range.
            - str: returns all values for that field as a 1D NDArray.
                   missing field returns np.full(len(self,), np.nan).
            - list[str]: returns a 2D NDArray of selected columns.
                   non-existing columns are filled with NaN (column-wise).
                   order is preserved as in the input list.

        Returns
        -------
        Dataset or NDArray
            - int/slice -> Dataset
            - str/list[str] -> np.ndarray

        Examples
        --------
        >>> from glide.core.dataset import Dataset
        >>> dataset = Dataset([{"y_true": i, "y_proxy": i % 2 == 0} for i in range(5)])
        >>> dataset.append({"y_true": 5})
        >>> dataset[0]
        [{'y_true': 0, 'y_proxy': True}]
        >>> dataset[0:3]
        [{'y_true': 0, 'y_proxy': True}, {'y_true': 1, 'y_proxy': False}, {'y_true': 2, 'y_proxy': True}]
        >>> dataset["y_true"]
        array([0., 1., 2., 3., 4., 5.])
        >>> dataset[["y_true", "y_proxy", "unknown"]]
        array([[ 0.,  1., nan],
               [ 1.,  0., nan],
               [ 2.,  1., nan],
               [ 3.,  0., nan],
               [ 4.,  1., nan],
               [ 5., nan, nan]])
        >>> dataset[dataset["y_proxy"] == True]
        [{'y_true': 0, 'y_proxy': True}, {'y_true': 2, 'y_proxy': True}, {'y_true': 4, 'y_proxy': True}]
        """

        # If key is a string return a column, else if an integer index return a record
        if isinstance(key, str):
            try:
                column = self.to_numpy([key])[:, 0]
                return column
            except ValueError:
                empty_column = np.full((len(self),), np.nan, dtype=float)
                return empty_column

        if isinstance(key, list) and all(isinstance(k, str) for k in key):
            # multiple columns
            try:
                multiple_existing_columns = self.to_numpy(key)
                return multiple_existing_columns
            except ValueError:
                # missing columns -> NaN columns
                missing_and_existing_columns = self._select_columns_with_nan(key)
                return missing_and_existing_columns

        if isinstance(key, slice):
            records_slice = Dataset(super().__getitem__(key))
            return records_slice

        if isinstance(key, int):
            try:
                record = Dataset([super().__getitem__(key)])
                return record
            except IndexError:
                return Dataset()

        if isinstance(key, np.ndarray) and key.dtype == bool:
            filtered_records = Dataset([r for i, r in enumerate(self) if key[i]])
            return filtered_records

        raise TypeError(f"Unsupported key type: {type(key)}")

    def _select_columns_with_nan(self, cols: List[str]) -> NDArray:
        """Build 2D array with known columns and NaN-filled missing columns.

        Parameters
        ----------
        cols : List[str]
            Requested column names.

        Returns
        -------
        NDArray
            2D array of shape (n_records, len(cols)) with NaN for missing columns.
        """
        all_columns = {k for record in self for k in record}
        known = [c for c in cols if c in all_columns]
        missing = [c for c in cols if c not in all_columns]

        arr_known = self.to_numpy(known) if known else np.empty((len(self), 0), dtype=float)
        arr_missing = np.full((len(self), len(missing)), np.nan, dtype=float)

        final_array = np.concatenate([arr_known, arr_missing], axis=1)

        return final_array

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
