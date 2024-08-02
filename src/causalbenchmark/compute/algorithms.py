"""
Various Causal Discovery Algorithm with unified API.
"""

# Standard 
from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np
import pandas as pd
from functools import wraps

# Own 
from ..util import pool_dfs, measure_time, same_columns
from . import ut_igsp

# Third party
from sempler.utils import dag_to_cpdag
from causallearn.search.ConstraintBased.PC import pc
import ges
import gies
import gnies
from notears.linear import notears_linear
from golempckg import fit_golem, postprocess
from CausalDisco.baselines import var_sort_regress, r2_sort_regress
import causalicp


#------------------------------------------------------
#------------------------------------------------------
# Helper

def _linear_to_binary(adj_mat: np.ndarray, var: list[str]):
    """
    Input[i,j]!=0 is transformed to Output[i,j]=1 and 
    index/columns are added.
    """
    adj_mat[adj_mat != 0] = 1
    return pd.DataFrame(adj_mat, index=var, columns=var)

def return_cpdag_if_wanted(flag_name):
    """Creates decorator which applies dag_to_cpdag if the instance variable `flag_name` is True."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            transform_bool = getattr(self, flag_name) # Get bool whether transformation is required
            if not isinstance(transform_bool, bool):
                raise ValueError(f"Instance variable '{flag_name}' must be True or False")
            dag_adj_mat_df = func(self, *args, **kwargs) # Get result of wrapped function
            if transform_bool is False:
                # No transformation to cpdag required
                return dag_adj_mat_df
            # Transformation to cpdag required
            try:
                # Apply transformation
                cpdag_np = dag_to_cpdag(dag_adj_mat_df.values)
                var = dag_adj_mat_df.columns
                return pd.DataFrame(cpdag_np, index=var, columns=var)
            except ValueError as e:
                print(f"Passed graph is not a valid DAG: {e}")
                return dag_adj_mat_df # Return original df on error     
        return wrapper
    return decorator


#------------------------------------------------------
#------------------------------------------------------
# Abstract base class

class Algorithm(ABC):
    """
    Abstract class where all Causal Inference algorithms inherit from.
    Each subclass encapuslates a third-party implementation of the respective
    algorithm and adheres to the unified API defined in this superclass.
    """

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
                adjacency matrix A with 
                - A[i,j]=1 and A[j,i] = 0 <-> directed edge from i to j
                - A[i,j]=1 and A[j,i] = 1 <-> undirected edge between i and j
                - A[i,j]=0 and A[j,i] = 0 <-> no edge between i and j
        """
        pass


#------------------------------------------------------
#------------------------------------------------------
# Algorithm implementations


#------------------------------------------------------
# Constraint based

class PC(Algorithm):
    """Encapsulates PC implementation from causal-learn package."""

    def __init__(self, alpha: float, indep_test= "fisherz"):
        """
        Initialize with wanted hyperparameters.

        Args:
            alpha (float): Significance level for the indep. test.
            indep_test (str, optional): The type of independence 
                test to use. Options are:
                - "fisherz": Fisher's Z conditional independence test
                - "chisq": Chi-squared conditional independence test
                - "gsq": G-squared conditional independence test
                - "kci": Kernel-based conditional independence test
                Defaults to "fisherz".
        """
        super().__init__(alg_name=self.__class__.__name__)
        self._alpha = alpha
        self._indep_test = indep_test

    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        """See superclass fit fct. for documentation."""
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
        """Turns causal-learn adj. matrix into format required by 'Algorithm' API."""
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


class UT_IGSP(Algorithm):
    """Encapsulates UT_IGSP implementation from causaldag package."""    
    def __init__(self,
                 alpha_ci: float, 
                 alpha_inv: float, 
                 debug=0, 
                 completion="gnies", 
                 test="hsic", 
                 obs_idx=0):
        """
        Initialize UT_IGSP object with desired hyperparameters.

        Args:
            alpha_ci (float): Level of conditional independence test.
            alpha_inv (float): Level of test that two Gaussians are equal.
            completion (str, optional): What equivalence class to compute
                based on UT-IGSP's output. Defaults to "gnies".
            test (str, optional): What test to use. Options: "hsic", "gauss".
                Defaults to "hsic".
            obs_idx (int, optional): The index of the observational data in the
                passed list of dataframes. Defaults to 0.
        """
        super().__init__(self.__class__.__name__)
        self._alpha_ci = alpha_ci
        self._alpha_inv = alpha_inv
        self._debug = debug
        self._completion = completion
        self._test = test
        self._obs_idx = obs_idx

    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        """See superclass fit fct. for documentation."""
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
# Score based

class GES(Algorithm):
    """Encapsulates GES implementation from ges package."""
    def __init__(self,
                 phases=['forward', 'backward', 'turning'],
                 iterate=False,
                 debug=0
                 ):
        """
        GES does not require any real hyperparameters.
        
        For implementation options, see documentation of ges.
        """
        super().__init__(alg_name=self.__class__.__name__)
        self._phases = phases
        self._iterate = iterate
        self._debug = debug

    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        """See superclass fit fct. for documentation."""
        pooled_data = pool_dfs(data)
        est_adj_mat, _ = ges.fit_bic(
            data=pooled_data.values,
            phases=self._phases,
            iterate=self._iterate,
            debug=self._debug
        )
        var = pooled_data.columns
        return pd.DataFrame(est_adj_mat, index=var, columns=var)


class GIES(Algorithm):
    """Encapsulates GIES implementation from gies package."""
    def __init__(self, 
                 interventions: list[list],
                 A0 = None,
                 phases = ['forward', 'backward', 'turning'], 
                 iterate = True, 
                 debug = 0):
        """
        Initialize with true interventions.

        Args:
            interventions (list[list]): One list of intervention targets
                (encoded as variable names) per dataset that will be passed.
            For implementation related parameters, see gies documentation.
        """
        super().__init__(alg_name=self.__class__.__name__)
        self._interventions=interventions
        self._A0 = A0
        self._phases = phases
        self._iterate = iterate
        self._debug = debug
    
    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        """See superclass fit fct. for documentation."""
        if not same_columns(data):
            raise ValueError("Not all passed dfs have the same columns.")
        if len(self._interventions) != len(data):
            raise ValueError("Data size differs from intervention size.")
        variables = list(data[0].columns)
        if not all(var in variables for inner_list in self._interventions for var in inner_list):
            raise ValueError("Unknown intervention targets")
        # Transform strings to indices 
        interventions = [[variables.index(var) for var in inner_list] for inner_list in self._interventions]
        estimate, _ = gies.fit_bic(
            data=[df.values for df in data], # Unpack dfs to np.ndarrays
            I=interventions,
            A0=self._A0,
            phases=self._phases,
            iterate=self._iterate,
            debug=self._debug
        )
        return pd.DataFrame(estimate, index=variables, columns=variables)


class GNIES(Algorithm):
    """Encapsulates GNIES implementation from gnies package."""
    def __init__(self, 
                lmbda=None,
                known_targets=set(),
                approach="greedy",
                I0=set(),
                phases=["forward", "backward"],
                direction="forward",
                center=True,
                ges_iterate=True,
                ges_phases=["forward", "backward", "turning"],
                debug=0
                ):
        """
        Initialize GNIES object. No hyperparameters necessary.

        For implemenation related parameters, see gnies documentation."""
        super().__init__(alg_name=self.__class__.__name__)
        self._lmbda = lmbda
        self._known_targets = known_targets
        self._approach = approach
        self._I0 = I0
        self._phases = phases
        self._direction = direction
        self._center = center
        self._ges_iterate = ges_iterate
        self._ges_phases = ges_phases
        self._debug = debug
    
    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        """See superclass fit fct. for documentation."""
        if not same_columns(data):
            raise ValueError("Not all passed dfs have the same columns.")
        _, estimate, _ = gnies.fit(
            data=[df.values for df in data],
            lmbda=self._lmbda,
            known_targets=self._known_targets,
            approach=self._approach,
            I0 = self._I0,
            phases=self._phases,
            direction=self._direction,
            center=self._center,
            ges_iterate=self._ges_iterate,
            ges_phases=self._ges_phases,
            debug=self._debug
        )
        var = data[0].columns
        return pd.DataFrame(estimate, index=var, columns=var)


#------------------------------------------------------
# Continuous optimization

class NoTears(Algorithm):
    """Encapsulates NoTears implementation from notears package."""
    def __init__(self,
                 return_cpdag: bool = False,
                 lambda1: float = 0.1,
                 loss_type: str = 'l2',
                 max_iter: int = 100,
                 h_tol: float = 1e-8,
                 rho_max: float = 10000000000000000,
                 w_threshold: float = 0.3
                 ):
        """
        Initialize NoTears object with desired hyperparameters.

        Args:
            return_cpdag (bool): Whether to transform NoTEARS output into 
                a CPAG. Defaults to False.
            lambda1 (float, optional): L1 penalty parameter for objective 
                function. Defaults to 0.1.
            loss_type (str, optional): Loss type employed in objective function.
                Defaults to 'l2'.
            For implemenation related parameters, see notears documentation.
        """
        super().__init__(alg_name=self.__class__.__name__)
        self._return_cpdag = return_cpdag
        self._lambda1 = lambda1
        self._loss_type = loss_type
        self._max_iter = max_iter
        self._h_tol = h_tol
        self._rho_max = rho_max
        self._w_threshold = w_threshold

    @measure_time
    @return_cpdag_if_wanted('_return_cpdag')
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        """See superclass fit fct. for documentation."""
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
    """Encapsulates Golem implementation from golempckg package."""    
    def __init__(self,
                equal_variances: bool,
                return_cpdag: bool = False,
                lambda_1: float = None,
                lambda_2: float = None,
                postproc_threshold: float = 0.3,
                num_iter=1e+5, 
                learning_rate=1e-3, 
                seed=1,
                checkpoint_iter=None,
                ):
        """
        Initialize Golem object with desired hyperparameters.

        Args:
            equal_variances (bool): Whether equal noise variances are assumed 
                when building the objective function. If True, objective function
                will be based on the MSE error, if False, objective function 
                will be bassed on gaussian likelihood.
            return_cpdag (bool): Whether to transform NoTEARS output into 
                a CPAG. Defaults to False.
            lambda_1 (float, optional): L1 penalty parameter. Defaults to 2e-2 
                or 5.0 depending on equal_variances.
            lambda_2 (float, optional): L2 penalty parameter. Defaults to 2r-3
                or 5.0 depending on equal_variances.
            postproc_threshold (float, optional): Edge weights below this threshold
                are set to 0 in postprocessing. Defaults to 0.3.
            For implemenation related parameters, see notears documentation.
        """
        super().__init__(alg_name=self.__class__.__name__)
        self._equal_variances = equal_variances
        self._return_cpdag = return_cpdag
        # Default parameters recommended in golem repository.
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
    @return_cpdag_if_wanted('_return_cpdag')
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        """See superclass fit fct. for documentation."""
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

#------------------------------------------------------
# Sortability based

class VarSortRegress(Algorithm):
    """Encapsulates VarSortRegress implementation from CausalDisco package."""    
    def __init__(self):
        """ No hyperparameters to pass for VarSortRegress."""
        super().__init__(alg_name=self.__class__.__name__)

    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        """See superclass fit fct. for documentation."""
        pooled_data = pool_dfs(data)
        adj_mat = var_sort_regress(X=pooled_data.values) # Returns linear adj. matrix
        var = pooled_data.columns
        return _linear_to_binary(adj_mat, var)


class R2SortRegress(Algorithm):
    """Encapsulates R2SortRegress implementation from CausalDisco package."""    
    def __init__(self):
        """ No hyperparameters to pass for R2SortRegress."""
        super().__init__(alg_name=self.__class__.__name__)

    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        """See superclass fit fct. for documentation."""
        pooled_data = pool_dfs(data)
        adj_mat = r2_sort_regress(X=pooled_data.values) # Returns linear adj. matrix
        var = pooled_data.columns
        return _linear_to_binary(adj_mat, var)


#------------------------------------------------------
# Invariance based

class ICP(Algorithm):
    """Encapsulates ICP implementation from causalicp package."""    
    def __init__(self,
                 target: str,
                 alpha: float = 0.05,
                 sets: list = None,
                 precompute: bool = True,
                 verbose: bool = False,
                 ):
        """
        Initialize ICP object with desired hyperparameters.

        Args:
            target (str): Which variable to predict. 
            alpha (float, optional): Significance level for the test 
                procedure. Defaults to 0.05.
            For implemenation related parameters, see causalicp documentation.
        """
        super().__init__(alg_name=self.__class__.__name__)
        self._target = target
        self._alpha = alpha
        self._sets = sets
        self._precompute = precompute
        self._verbose = verbose
    
    @measure_time
    def fit(self, data: Iterable[pd.DataFrame]) -> list[pd.DataFrame, float]:
        """See superclass fit fct. for documentation."""
        if not same_columns(data):
            raise ValueError("Not all passed dfs have the same columns.")
        if not self._target in list(data[0].columns):
            raise ValueError("Passed Target does not exist in data.")
        target_index = list(data[0].columns).index(self._target)

        icp_fit = causalicp.fit(
            data=[df.values for df in data], # Unpack pd.dfs to np.ndarrays
            target=target_index,
            alpha=self._alpha,
            sets=self._sets,
            precompute=self._precompute,
            verbose=self._verbose
        )

        parents = icp_fit.estimate # None if estimated parent set is empty
        if parents is not None:
            edges = [(parent, target_index) for parent in parents] # Create edges
        else:
            edges = []

        return self._transform_to_adj_mat(edges, data[0].columns)
    
    def _transform_to_adj_mat(self, edges: list, var: list[str]) -> pd.DataFrame:
        """
        Turns list of edges into Adjacency matrix.

        Args:
            edges (list): List of edges. Variables encoded by their index.
            var (list[str]): List of variables the indices relate to.

        Returns:
            pd.DataFrame: Adjacency matrix in format desired by 'Algorithm'
                class API.
        """
        adj_matrix = pd.DataFrame(0, index=var, columns=var)
        if len(edges)==0:
            return adj_matrix
        # Unpack list of edges into list with row indices and list with column indices
        rows, cols = zip(*edges) 
        adj_matrix.iloc[rows, cols] = 1
        return adj_matrix
        


