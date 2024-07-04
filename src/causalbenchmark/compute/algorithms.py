# Standard 
from abc import ABC, abstractmethod
from typing import Iterable
import pandas as pd

# Own 
from .dictable import Dictable
from ..util import pool_dfs, measure_time

# Third party
import ges

#------------------------------------------------------
# Abstract base class

class Algorithm(ABC):
    """Abstract class where all Causal Inference algorithms inherit from"""

    def __init__(self, alg_name: str):
        """Used to store hyperparameters"""
        self._alg_name = alg_name
    
    @abstractmethod
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        """
        Fits the respective algorithm to the passed data.

        Args:
            data (tuple[pd.DataFrame]): The data to fit the algorithm to.
                Each DataFrame has variables as columns and observations as 
                rows. The column names must be equal and in the same order 
                across all DataFrames.

        Returns:
            tuple[float, pd.DataFrame]: Runtime and estimated binary 
                adjacency matrix A with A[i,j]=1 <-> edge from i to j
        """
        pass


#------------------------------------------------------
# Algorithm implementations

class PC(Algorithm):
    # Use Causal-Learn implementation
    pass

class GES(Algorithm):

    def __init__(self):
        """ No hyperparameters to pass for GES. """
        super().__init__(
            alg_name=self.__class__.__name__
        )

    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        pooled_data = pool_dfs(data)
        est_adj_mat, _ = ges.fit_bic(
            data=pooled_data.values,
            phases=['forward', 'backward', 'turning'],
            debug=0
        )
        est_adj_mat_df = pd.DataFrame(
            data=est_adj_mat, 
            index=pooled_data.columns,
            columns=pooled_data.columns
        )
        return est_adj_mat_df

class NoTears(Algorithm):
    pass 

class Golem(Algorithm):
    pass

class VarSortRegress(Algorithm):
    pass 

class R2SortRegress(Algorithm):
    pass 

class ICP(Algorithm):
    pass 


