# Standard 
from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np
import pandas as pd

# Own 
from .dictable import Dictable
from ..util import pool_dfs, measure_time, same_columns
from . import ut_igsp

# Third party
from causallearn.search.ConstraintBased.PC import pc
import ges
from notears.linear import notears_linear
from golempckg import fit_golem, postprocess
from CausalDisco.baselines import var_sort_regress, r2_sort_regress
import causalicp

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

    def __init__(self, alpha: float, indep_test= "fisherz"):
        super().__init__(alg_name=self.__class__.__name__)
        self._alpha = alpha
        self._indep_test = indep_test

    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        pooled_data = pool_dfs(data)
        pc_model = pc(
            data=pooled_data.values,
            alpha=self._alpha,
            indep_test=self._indep_test,
            show_progress=False
        )
        pc_graph = pc_model.G.graph
        var = data[0].columns
        return self._transform_to_adj_mat(pc_graph, var)

    def _transform_to_adj_mat(self, pc_graph: np.ndarray, var: list[str]):
        adj_matrix = np.zeros_like(pc_graph)

        # Undirected Edge between i and j
        # In PC graph: A[i,j] = A[j,i] = -1
        # In our graph: A[i,j] = A[j,i] = 1
        undirected_edges = (pc_graph == -1) & (pc_graph.T == -1)
        adj_matrix[undirected_edges] = 1
        
        # Bidirectional Edge between i and j
        # In PC graph: A[i,j]= A[j,i] = 1
        # In our graph: A[i,j] = A[j,i] = 1
        bidirectional_edges = (pc_graph == 1) & (pc_graph.T == 1)
        adj_matrix[bidirectional_edges] = 1
        bidirectional_edges_nr = np.count_nonzero(bidirectional_edges)
        if bidirectional_edges_nr > 0:
            print(f"Bidirectional edges detected: {bidirectional_edges_nr}")

        # Directed Edge from i to j
        # In PC graph: A[i,j] = -1, A[j,i] = 1
        # In our graph: A[i,j] = 1, A[j,i] = 0
        directed_edges = (pc_graph == -1) & (pc_graph.T == 1)
        adj_matrix[directed_edges] = 1

        return pd.DataFrame(adj_matrix, index=var, columns=var)


class GES(Algorithm):

    def __init__(self):
        """ No hyperparameters to pass for GES. """
        super().__init__(alg_name=self.__class__.__name__)

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

    def __init__(self,
                 lambda1: float = 0.1,
                 loss_type: str = 'l2',
                 max_iter: int = 100,
                 h_tol: float = 1e-8,
                 rho_max: float = 10000000000000000,
                 w_threshold: float = 0.3
                 ):
        super().__init__(alg_name=self.__class__.__name__)
        self._lambda1 = lambda1
        self._loss_type = loss_type
        self._max_iter = max_iter
        self._h_tol = h_tol
        self._rho_max = rho_max
        self._w_threshold = w_threshold

    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        pooled_data = pool_dfs(data)
        notears_graph = notears_linear(
            X=pooled_data.values,
            lambda1=self._lambda1,
            loss_type=self._loss_type,
            max_iter=self._max_iter,
            h_tol=self._h_tol,
            rho_max=self._rho_max,
            w_threshold=self._w_threshold
        )
        var = pooled_data.columns
        return _linear_to_binary(notears_graph, var)
    


class Golem(Algorithm):
    
    def __init__(self,
                equal_variances: bool,
                postproc_threshold = 0.3,
                lambda_1 = None,
                lambda_2 = None,
                num_iter=1e+5, 
                learning_rate=1e-3, 
                seed=1,
                checkpoint_iter=None,
                ):
        
        super().__init__(alg_name=self.__class__.__name__)
        self._equal_variances = equal_variances
        if equal_variances and lambda_1 is None and lambda_2 is None:
            self._lambda_1=2e-2, 
            self._lambda_2=5.0
        elif not equal_variances and lambda_1 is None and lambda_2 is None:
            self._lambda_1=2e-3, 
            self._lambda_2=5.0
        else:
            raise ValueError("GOLEM not correctly initialized.")
        self._postproc_threshold = postproc_threshold
        self._num_iter = num_iter
        self._learning_rate = learning_rate
        self._seed = seed
        self._checkpoint_iter = checkpoint_iter
    
    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        pooled_data = pool_dfs(data)
        adj_mat_raw = fit_golem(
            X=pooled_data.values,
            lambda_1=self._lambda_1,
            lambda_2=self._lambda_2,
            equal_variances=self._equal_variances,
            num_iter=self._num_iter,
            learning_rate=self._learning_rate,
            seed=self._seed,
            checkpoint_iter=self._checkpoint_iter
        )
        adj_mat = postprocess(B=adj_mat_raw, graph_thres=self._postproc_threshold)
        var = pooled_data.columns
        return _linear_to_binary(adj_mat, var)



class VarSortRegress(Algorithm):
    
    def __init__(self):
        """ No hyperparameters to pass for VarSortRegress."""
        super().__init__(alg_name=self.__class__.__name__)

    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        pooled_data = pool_dfs(data)
        adj_mat = var_sort_regress(X=pooled_data.values)
        var = pooled_data.columns
        return _linear_to_binary(adj_mat, var)


class R2SortRegress(Algorithm):

    def __init__(self):
        """ No hyperparameters to pass for VarSortRegress."""
        super().__init__(alg_name=self.__class__.__name__)

    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        pooled_data = pool_dfs(data)
        adj_mat = r2_sort_regress(X=pooled_data.values)
        var = pooled_data.columns
        return _linear_to_binary(adj_mat, var)

class ICP(Algorithm):

    def __init__(self,
                 target: str,
                 alpha: float = 0.05,
                 sets: list = None,
                 precompute: bool = True,
                 verbose: bool = False,
                 ):
        super().__init__(alg_name=self.__class__.__name__)
        self._target = target
        self._alpha = alpha
        self._sets = sets
        self._precompute = precompute
        self._verbose = verbose
    
    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        if not same_columns(data):
            raise ValueError("Not all passed dfs have the same columns.")
        if not self._target in list(data[0].columns):
            raise ValueError("Passed Target does not exist in data.")
        target_index = list(data[0].columns).index(self._target)

        icp_fit = causalicp.fit(
            data=[df.values for df in data],
            target=target_index,
            alpha=self._alpha,
            sets=self._sets,
            precompute=self._precompute,
            verbose=self._verbose
        )

        parents = icp_fit.estimate # None if estimated parent set is empty
        if parents is not None:
            edges = [(parent, target_index) for parent in parents]
        else:
            edges = []

        return self._transform_to_adj_mat(edges, data[0].columns)
    
    def _transform_to_adj_mat(self, edges: list, var: list[str]):
        adj_matrix = pd.DataFrame(0, index=var, columns=var)
        if len(edges)==0:
            return adj_matrix
        rows, cols = zip(*edges)
        adj_matrix.iloc[rows, cols] = 1
        return adj_matrix
        
class UT_IGSP(Algorithm):

    def __init__(self,
                 alpha_ci, 
                 alpha_inv, 
                 debug=0, 
                 completion="gnies", 
                 test="hsic", 
                 obs_idx=0):
        super().__init__(self.__class__.__name__)
        self._alpha_ci = alpha_ci
        self._alpha_inv = alpha_inv
        self._debug = debug
        self._completion = completion
        self._test = test
        self._obs_idx = obs_idx

    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        
        if not same_columns(data):
            raise ValueError("Not all passed dfs have the same columns.")
        ut_fit = ut_igsp.fit(
            data=[df.values for df in data],
            alpha_ci=self._alpha_ci,
            alpha_inv=self._alpha_inv,
            debug=self._debug, 
            completion=self._completion,
            test=self._test,
            obs_idx=self._obs_idx
        )
        fitted_icpdag = ut_fit[0] # Get ICPDAG
        var = data[0].columns
        return pd.DataFrame(fitted_icpdag, index=var, columns=var)


#------------------------------------------------------
# Helper

def _linear_to_binary(adj_mat: np.ndarray, var: list[str]):
    adj_mat[adj_mat != 0] = 1
    return pd.DataFrame(adj_mat, index=var, columns=var)
