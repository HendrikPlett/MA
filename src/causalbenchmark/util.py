"""
Util functions for the causalbenchmark package.
"""


# Standard 
import pandas as pd 
from typing import Iterable, Union
import time
from functools import wraps


#------------------------------------------------------
# Decorators

def measure_time(func):
    """Decorator to measure the runtime"""
    @wraps(func)
    def wrapper_fct(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        return (result, runtime)
    return wrapper_fct


#------------------------------------------------------
# List operations

def has_duplicates(l: list) -> bool:
    """Whether or not a list contains an element more than once."""
    return len(l) != len(set(l))

def enforce_no_duplicates(l_of_l: list[list]):
    """Raises a ValueError if any passed list contains dublicates."""
    if any([has_duplicates(l) for l in l_of_l]):
        raise ValueError("Duplicates in at least one passed list.")

def give_superlist(first: list, second: list) -> list:
    """
    Returns the list that contains the other list.

    Args:
        first (list): A list without dublicates (will also be checked).
        second (list): A list without dublicates (will also be checked).

    Raises:
        ValueError: If no list contains the other list.

    Returns:
        list: The list that contains the other list.
    """
    enforce_no_duplicates([first, second])
    first_set = set(first)
    second_set = set(second)
    if first_set.issuperset(second_set):
        return first
    elif second_set.issuperset(first_set):
        return second
    else:
        raise ValueError("No Variable list contains the other.")

def give_sublist(first: list, second: list) -> list:
    """
    Returns the list that is contained by the other list.

    Args:
        first (list): A list without dublicates (will also be checked).
        second (list): A list without dublicates (will also be checked).

    Raises:
        ValueError: If no list is contained by the other list.

    Returns:
        list: The list that is contained by the other list.
    """
    enforce_no_duplicates([first, second])
    first_set = set(first)
    second_set = set(second)
    if first_set.issubset(second_set):
        return first
    elif second_set.issubset(first_set):
        return second
    else:
        raise ValueError("No Variable list contains the other.")

def variables_increase(first: list, second: list) -> bool:
    """
    Whether 'first' has a subset of variables compared to 'second'.
    Also implicitly checks for dublicates.
    """
    return (first == give_sublist(first, second))
    
def same_order(first: list, second: list) -> bool:
    """
    Whether the common elements in two lists have the same order.
    """
    superlist = give_superlist(first, second)
    sublist = give_sublist(first, second)
    superlist_reduced = [el for el in superlist if el in sublist]
    return superlist_reduced == sublist


#------------------------------------------------------
# Data Dataframe operations

def standardize_dfs(dfs: Iterable[pd.DataFrame]) -> list[pd.DataFrame]:
    """Standardizes each column in each passed df to (mean, sd)=(0,1)."""
    stand_dfs = []
    for df in dfs:
        mean = df.mean()
        sd = df.std()
        sd.replace(0,1) # Handle case of 0 standard deviation
        stand_dfs.append((df-mean)/sd)
    return stand_dfs

def pool_dfs(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates DataFrames with same column names and ordering
        along the indices. Indices will be reset. 

    Args:
        dfs (Iterable): Iterable of DataFrames with same column 
            names and ordering (will be checked).

    Raises:
        ValueError: If the input dfs columns' are not the same.

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

    Raises:
        ValueError: If the columns are not the same.

    Returns:
        bool: Whether all columns are equal and in the same order.
    """
    if len(dfs)==0:
        raise ValueError("No DataFrames were passed.")
    ref_columns = list(dfs[0].columns)
    same_names_and_order = all(df.columns.tolist() == ref_columns for df in dfs)
    return same_names_and_order
    

def bootstrap_sample(datasets: Iterable[pd.DataFrame], sample_sizes: Iterable[Union[int, float]], 
                     seed: int) -> list[pd.DataFrame]:
    """
    Generates a single bootstrap sample from the passed list of dataframes.

    Args:
        datasets (Iterable[pd.DataFrame]): Data to bootstrap from.
        sample_sizes (Iterable[Union[int, float]]): A sample size for each df.
            Either a percentage (float) or number of samples (int).
        seed (int): Random seed. Note that different seeds will be used for the
            sampling of each passed df.

    Raises:
        ValueError: If the sample sizes have an incorrect format.

    Returns:
        list[pd.DataFrame]: Bootstrap sample.
    """

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
# Adjacency matrix dataframe operations

def is_sub_adj_mat(df: pd.DataFrame, larger_df: pd.DataFrame) -> bool:
    """
    Whether df has a subset of variables compared to larger_df.

    Args:
        df (pd.DataFrame): Valid adjacency matrix (will be checked).
        larger_df (pd.DataFrame): Valid potentially larger adjacency 
            matrix (will be checked).

    Returns:
        bool: Whether df has a subset of variables compared to larger_df
    """
    enforce_valid_adj_mat(df)
    enforce_valid_adj_mat(larger_df)
    if variables_increase(df.index.to_list(), larger_df.index.to_list()):
        return True
    else:
        return False

def enforce_sub_adj_mat(df: pd.DataFrame, larger_df: pd.DataFrame):
    """Raises ValueError if df is not a sub adjacency matrix of larger_df."""
    if not is_sub_adj_mat(df, larger_df):
        raise ValueError("df is not sub matrix of larger_df.")

def reduce_to_size(df: pd.DataFrame, reduce_to: pd.DataFrame) -> pd.DataFrame:
    """
    Removes all rows/cols from df that are not in reduce_to.

    Args:
        df (pd.DataFrame): Valid adjacency matrix (will be checked).
        reduce_to (pd.DataFrame): Valid sub adjacency matrix of df 
            (will be checked).

    Returns:
        pd.DataFrame: df reduced to reduce_to.
    """
    enforce_valid_adj_mat(df)
    enforce_valid_adj_mat(reduce_to)
    enforce_sub_adj_mat(reduce_to, df)
    common_var = df.columns.intersection(reduce_to.columns)
    df_reduced = df.loc[common_var, common_var]
    return df_reduced

def pad_zeros_to_size(df: pd.DataFrame, pad_to: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rows/columns with zeros to df s.t. it has the same variables as pad_to.

    Args:
        df (pd.DataFrame): Valid sub adjacency matrix of pad_to (will be checked).
        pad_to (pd.DataFrame): Valid adjacency matrix (will be checked).

    Returns:
        pd.DataFrame: Padded df.
    """
    enforce_valid_adj_mat(df)
    enforce_valid_adj_mat(pad_to)
    enforce_sub_adj_mat(df, pad_to)
    df_pad = df.reindex(
        index=pad_to.index,
        columns=pad_to.columns,
        fill_value=0)
    return df_pad


def enforce_valid_bstr_adj_mat(graph: pd.DataFrame):
    """
    Checks that all elements in the matrix are in [0,1], i.e.
        valid percentages after averaging over DAGs generated from 
        bootstrap samples.
    """
    enforce_valid_adj_mat(graph)
    if not ((graph >= 0) & (graph <= 1)).all().all():
        raise ValueError("Entries outside of [0,1].")
    
def enforce_binary_adj_mat(graph: pd.DataFrame):
    """Raieses ValueError if passed graph contains non-binary values."""
    enforce_valid_adj_mat(graph)
    if not graph.isin([0,1]).all().all():
        raise ValueError("Not all entries binary.")

def enforce_valid_adj_mat(graph: pd.DataFrame):
    """
    Raises ValueError if
    - graph has different index and column names or
    - graph has dublicate column names.
    """
    if not same_columns([graph, graph.transpose()]):
        raise ValueError("Different index/column names.")
    enforce_no_duplicates([graph.columns.to_list()])

    
