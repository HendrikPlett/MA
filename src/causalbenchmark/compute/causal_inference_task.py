"""
Functionality to:
    - Compute all and the average consistent extensions from the 'Algorithm' output.
    - Compute sortability of passed data.
"""

# Standard 
from typing import Iterable
import numpy as np
import pandas as pd

# Third party
from CausalDisco.analytics import var_sortability, r2_sortability
from sempler.utils import all_dags

# Own
from .algorithms import Algorithm
from ..util import same_columns, pool_dfs

class CausalInferenceTask:
    """
    Analyze the data that is passed to 'Algorithm' and 
    postprocess 'Algorithm's output.
    """
    def __init__(self, 
                 algorithm: Algorithm,
                 data: Iterable[pd.DataFrame], 
                 true_dag: pd.DataFrame
                 ):
        """
        Passed arguments cannot be changed later on.

        Args:
            algorithm (Algorithm): The algorithm to use.
            data (Iterable[pd.DataFrame]): The data to pass to the algorithm.
            true_dag (pd.DataFrame): The DAG according to which the passed 
                data was generated.
        """
        # --- Check validity of input
        if not same_columns((*data, true_dag, true_dag.transpose())):
            raise ValueError("Different variables are used in the data and/or TrueDag.")
        
        # --- Provided
        self._algorithm = algorithm
        self._data = data
        self._true_dag = true_dag
        # --- Computed later
        # Algorithm output
        self._estimated_graph = None
        self._runtime = None
        # Consistent extensions
        self._all_cons_extensions = None
        self._average_cons_extension = None 
        # Failure counter
        self._no_cons_extensions = False
        self._algorithm_crashed = False
        # Sortability
        self._var_sort = None
        self._r2_sort = None

    def run_task(self):
        """ 
        Computes sortability metrics, fits the passed algorithm
            to the passed data and computes the average consistent
            extension of the fitted PDAG.
        """
        self._compute_sortability()
        try:
            self._estimated_graph, self._runtime = self._algorithm.fit(
                data=self._data
            )
        except Exception as e:
            print(f"Exception thrown while fitting the algorithm: {e}")
            self._algorithm_crashed = True
            var = list(self._true_dag.columns)
            dim = len(var)
            # Set values for algorithm not to crash. 
            self._estimated_graph = pd.DataFrame(np.zeros((dim, dim)), columns=var, index=var)
            self._runtime = 0
        self._consistent_extensions()

        return self # Enables multiprocessing
        

    def get_estimated_graph(self) -> pd.DataFrame:
        """ Gets the estimated graph. """
        return self._estimated_graph
    
    def get_runtime(self) -> float:
        """ Gets the runtime needed to fit the algorithm. """
        return self._runtime
    
    def get_all_cons_extensions(self) -> list[np.ndarray]:
        """ Gets all consistent extensions of the estimated graph. """
        return self._all_cons_extensions

    def get_average_cons_extension(self) -> pd.DataFrame:
        """ Gets the average of all consistent extensions. """
        return self._average_cons_extension
    
    def get_no_consistent_extensions_flag(self) -> bool:
        """ Gets whether the returned pdag has no consistent extensions. """
        return self._no_cons_extensions

    def get_algorithm_crashed_flag(self) -> bool:
        """ Gets whether the causal discovery algorithm crashed while running. """
        return self._algorithm_crashed

    def get_var_sort(self) -> float:
        """ Gets the Variance Sortability of the passed dataset. """
        return self._var_sort
    
    def get_r2_sort(self) -> float:
        """ Gets the R2 Sortability of the passed dataset. """
        return self._r2_sort
    
    def _compute_sortability(self):
        """ Computes and saves Variance and R2 Sortability of the passed dataset. """
        try:
            self._var_sort = var_sortability(
                X=pool_dfs(self._data).values, 
                W=self._true_dag.values
            )
        except ValueError as e: # Can happen if data contains too many columns with 0 variance 
            self._var_sort = 0
        except Exception as e:
            self._var_sort = 0
            print(f"Exception thrown while computing var sortability: {e}")
        try:
            self._r2_sort = r2_sortability(
                X=pool_dfs(self._data).values, 
                W=self._true_dag.values
            )
        except ValueError as e: # Can happen if data contains too many columns with 0 variance
            self._r2_sort = 0
        except Exception as e:
            self._r2_sort = 0
            print(f"Exception thrown while camputing r2 sortability: {e}")
    
    def _consistent_extensions(self):
        """ 
        Computes and saves all consistent extensions of the fitted graph
            and computes and saves the average of these extensions.
        """
        self._all_cons_extensions = all_dags(self._estimated_graph.values).tolist()
        # Handle the case of zero valid consistent extensions
        if len(self._all_cons_extensions) == 0: 
            self._no_cons_extensions = True
            self._average_cons_extension = None
        else: 
            avg_dag = np.average(self._all_cons_extensions, axis=0)
            self._average_cons_extension = pd.DataFrame(
                data=avg_dag,
                index=self._estimated_graph.index,
                columns=self._estimated_graph.columns
            )

