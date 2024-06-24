import hashlib
from typing import Any, Dict, List, Iterable, Callable, Set, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_integer_dtype,
    is_float_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

HASH_FUNCTION: Callable = hashlib.sha256
# Used as a separator when concating fields for hashing, ensures there are no
# mistakes if fields are empty.
HASH_FIELD_SEPARATOR: str = ":"
COLLECTION_VARIABLE_TYPES = (
    list,
    set,
    tuple,
    np.ndarray,
)


def hash_string_iterable(string_iterable: Iterable[str]) -> str:
    input_str = HASH_FIELD_SEPARATOR.join(string_iterable)
    return HASH_FUNCTION(input_str.encode("utf-8")).hexdigest()


class DFValueHasher:
    def __init__(
        self,
        df: pd.DataFrame,
        columns_to_hash: List[str] = None,
        float_precision: int = 5,
    ):
        self.df = df.copy(deep=True)
        self.float_precision = float_precision
        if columns_to_hash is None or len(columns_to_hash) == 0:
            self.columns_to_hash = list(df.columns)
        else:
            self.columns_to_hash = columns_to_hash

    def __to_list(self, x: Union[List, Tuple, Set, np.ndarray]):
        if isinstance(x, np.ndarray):
            list_repr = x.tolist()
        else:
            # FIXME(pcyin): This can throw an error on non-iterable types.
            list_repr = list(x)
        return list_repr

    def _prep_series_categorical(self, series: pd.Series):
        if (
            is_object_dtype(series.dtype) or is_integer_dtype(series.dtype)
        ) and series.nunique() / len(series) < 0.5:
            return pd.factorize(series)[0]
        else:
            return self._prep_series(series)

    def _prep_series(self, series: pd.Series):
        dtype = series.dtype

        if is_numeric_dtype(dtype):
            series = series.fillna(0)
        elif is_object_dtype(dtype):
            series = series.fillna("")

        if is_integer_dtype(dtype):
            return series.fillna(0).astype(str)
        elif is_float_dtype(dtype):
            round_num = self.float_precision
            return series.apply(lambda x: round(x, round_num)).fillna(0).astype(str)
        elif dtype == bool:
            return series.astype(str)
        elif dtype == object:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    series = pd.to_datetime(series, errors="raise")
                    dtype = series.dtype
            except (ValueError, TypeError):
                try:
                    if series.nunique() / len(series) < 0.5:
                        return series.astype("category").astype(str)
                    else:
                        return series.astype(str)
                except Exception as e:
                    # * could handle containers within the column
                    return series.astype(str)

        if is_datetime64_any_dtype(dtype):
            try:
                return series.dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                series_apply = pd.to_datetime(series, errors="coerce")
                return series_apply.dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return series.astype(str)

    def prep_df(self, df: pd.DataFrame, prep_categorical: bool = False):
        for col in self.columns_to_hash:
            if prep_categorical:
                df[col] = self._prep_series_categorical(df[col])
            else:
                df[col] = self._prep_series(df[col])
        return df

    def hash(self) -> Dict[str, str]:
        df = self.prep_df(self.df)
        hashed_series = df[self.columns_to_hash].astype(str).apply(hash_string_iterable)
        return hashed_series.to_dict()

    def hash_categorical(self) -> Dict[str, str]:
        df = self.prep_df(self.df, prep_categorical=True)
        hashed_series = df[self.columns_to_hash].astype(str).apply(hash_string_iterable)
        return hashed_series.to_dict()
