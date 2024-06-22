# Standard 
import pandas as pd 
from typing import Iterable
import time


def measure_time(func):
    """Decorator to measure the runtime"""
    def wrapper_fct(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        return (result, runtime)
    return wrapper_fct


def pool_dfs(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates DataFrames with same column names and ordering
        along the indices. Indices will be reset. 

    Args:
        dfs (Iterable): Iterable of DataFrames with same column 
            names and ordering. 

    Returns:
        pd.DataFrame: The concatened DataFrame.
    """
    if not same_columns(dfs):
        raise ValueError("Dfs have different columns.")
    conc_dfs = pd.concat(dfs, axis=0, ignore_index=True)
    return conc_dfs 


def same_columns(dfs: Iterable[pd.DataFrame]) -> bool:
    """
    Checks if all passed dfs have the same columns in the same order.

    Args:
        dfs (Iterable): Iterable of pandas DataFrames.

    Returns:
        bool: Whether all columns are equal and in the same order.
    """
    if len(dfs)==0:
        raise ValueError("No DataFrames were passed.")
    ref_columns = list(dfs[0].columns)
    same_names_and_order = all(df.columns.tolist() == ref_columns for df in dfs)
    return same_names_and_order