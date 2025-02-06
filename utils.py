import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype

def nan_replace_t(t: pd.DataFrame):
    assert isinstance(t, pd.DataFrame)
    for v in t.columns:
        if is_numeric_dtype(t[v]):
            t.fillna({v: t[v].mean()}, inplace=True)
        else:
            t.fillna({v: t[v].mode()[0]}, inplace=True)