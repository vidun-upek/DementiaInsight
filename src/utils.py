import numpy as np
import pandas as pd

def to_numeric(df):
    na_values = [-4, 9, 88, 99, 888, 999, 9999] 
    return df.apply(pd.to_numeric, errors='coerce').replace(na_values, np.nan)

def to_string(df):
    return df.astype(str)
