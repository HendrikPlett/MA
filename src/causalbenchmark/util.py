# Standard 
import pandas as pd 
from typing import Iterable, Union
import time


#------------------------------------------------------
# Decorators

def measure_time(func):
    """Decorator to measure the runtime"""
    def wrapper_fct(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        return (result, runtime)
    return wrapper_fct


#------------------------------------------------------
# Dataframe operations

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


def bootstrap_sample(datasets: Iterable[pd.DataFrame], sample_sizes: Iterable[Union[int, float]], seed):

    assert len(datasets) == len(sample_sizes)
    
    bstr_spl = []
    counter = 0
    for dataset, sample_size in zip(datasets, sample_sizes):
        if 0 < sample_size <= 1.0:
            bstr_spl.append(dataset.sample(frac=sample_size, replace=True, random_state=seed+counter))
        elif sample_size > 1:
            bstr_spl.append(dataset.sample(n=sample_size, replace=True, random_state=seed+counter))
        else: 
            raise ValueError("The sample sizes have the wrong format.")
        counter += 1

    return bstr_spl


#------------------------------------------------------
# List operations

def give_superlist(first: list, second: list):
    first_set = set(first)
    second_set = set(second)
    if first_set.issuperset(second_set):
        return first
    elif second_set.issuperset(first_set):
        return second
    else:
        raise ValueError("No Variable list contains the other")

def give_sublist(first: list, second: list):
    first_set = set(first)
    second_set = set(second)
    if first_set.issubset(second_set):
        return first
    elif second_set.issubset(first_set):
        return second
    else:
        raise ValueError("No Variable list contains the other")

def same_order(first: list, second: list):
    superlist = give_superlist(first, second)
    sublist = give_sublist(first, second)
    superlist_reduced = [el for el in superlist if el in sublist]
    return superlist_reduced == sublist