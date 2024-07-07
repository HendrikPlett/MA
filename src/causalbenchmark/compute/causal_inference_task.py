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
    
    def __init__(self, 
                 algorithm: Algorithm,
                 data: Iterable[pd.DataFrame], 
                 true_dag: pd.DataFrame
                 ):
        
        # --- Check validity of input
        if not same_columns((*data, true_dag, true_dag.transpose())):
            raise ValueError("Different variables are used in the data and/or TrueDag.")
        
        # --- Provided
        self._algorithm = algorithm
        self._data = data
        self._true_dag = true_dag
        # --- Computed later
        self._estimated_graph = None
        self._runtime = None
        self._all_cons_extensions = None
        self._average_cons_extension = None 
        self._var_sort = None
        self._r2_sort = None

    def run_task(self):
        """ 
        Computes sortability metrics, fits the passed algorithm
            to the passed data and computes the average consistent
            extension of the fitted PDAG.
        """
        self._compute_sortability()
        self._estimated_graph, self._runtime = self._algorithm.fit(
            data=self._data
        )
        self._consistent_extensions()
        

    def get_estimated_graph(self) -> pd.DataFrame:
        """ Gets the estimated graph. """
        return self._estimated_graph
    
    def get_runtime(self) -> float:
        """ Gets the runtime needed to fit the graph. """
        return self._runtime
    
    def get_all_cons_extensions(self) -> list[np.ndarray]:
        """ Gets all consistent extensions of the estimated graph. """
        return self._all_cons_extensions

    def get_average_cons_extension(self) -> pd.DataFrame:
        """ Gets the average of all consistent extensions. """
        return self._average_cons_extension

    def get_var_sort(self) -> float:
        """ Gets the Variance Sortability of the passed dataset. """
        return self._var_sort
    
    def get_r2_sort(self) -> float:
        """ Gets the R2 Sortability of the passed dataset. """
        return self._r2_sort
    
    def _compute_sortability(self):
        """ Computes Variance and R2 Sortability of the passed dataset. """
        self._var_sort = var_sortability(
            X=pool_dfs(self._data).values, 
            W=self._true_dag.values
        )
        self._r2_sort = r2_sortability(
            X=pool_dfs(self._data).values, 
            W=self._true_dag.values
        )
    
    def _consistent_extensions(self):
        """ 
        Computes+saves all consistent extensions of the fitted graph
            and computes+saves the average of these extensions 
        """
        self._all_cons_extensions = all_dags(self._estimated_graph.values).tolist()
        if len(self._all_cons_extensions) == 0:
            print(f"Nr cons extensions: {len(self._all_cons_extensions)}")
            avg_dag = np.zeros_like(self._true_dag.values)
        else: 
            avg_dag = np.average(self._all_cons_extensions, axis=0)
        self._average_cons_extension = pd.DataFrame(
            data=avg_dag,
            index=self._estimated_graph.index,
            columns=self._estimated_graph.columns
        )

