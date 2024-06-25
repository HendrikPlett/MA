# Standard 
from typing import Iterable
import numpy as np
import pandas as pd

# Third party

# Own
from .algorithms import Algorithm
from ..util import same_columns, bootstrap_sample, give_superlist, same_order
from .causal_inference_task import CausalInferenceTask

class Bootstrap():

    def __init__(self, 
                 bootstrap_name: str, 
                 algorithm: Algorithm,
                 data_to_bootstrap_from: Iterable[pd.DataFrame], 
                 sample_sizes: tuple, 
                 nr_bootstraps: int,
                 true_dag: pd.DataFrame):
        
        # --- Check validity of input
        assert len(data_to_bootstrap_from)>=1, "No data passed"
        if not same_columns((*data_to_bootstrap_from, true_dag, true_dag.transpose())):
            raise ValueError("Different variables used in data and/or true_dag.")
        if not len(data_to_bootstrap_from) == len(sample_sizes):
            raise ValueError("sample_size and data_to_bootstrap_from must have same length.")
        assert nr_bootstraps >= 1, "nr_bootstraps must be an integer value larger than 1"
        assert all((isinstance(size, float) and 0<size<=1) or (isinstance(size, int) and size >1) 
                   for size in sample_sizes), "Sample sizes must be a float between 0 and 1 or an integer"
        
        # --- Provided
        self._bootstrap_name = bootstrap_name
        self._algorithm = algorithm 
        self._data_to_bootstrap_from = data_to_bootstrap_from
        self._sample_sizes = sample_sizes
        self._nr_bootstraps = nr_bootstraps
        self._true_dag = true_dag
        # --- Provided implicitly
        self._bootstrap_variables = true_dag.columns.to_list()
        # --- Computed later
        self._causal_inference_tasks = None
        self._avg_avg_cons_extension = None
        self._avg_runtime = None
        self._avg_var_sort = None
        self._avg_r2_sort = None

    def run_bootstrap(self):
        self._create_causal_inference_tasks()
        self._run_causal_inference_tasks()
        self._compute_averages()
    
    def get_bootstrap_variables(self) -> list:
        return self._bootstrap_variables

    def get_true_dag(self) -> pd.DataFrame:
        return self._true_dag
    
    def get_avg_avg_cons_extension(self) -> pd.DataFrame:
        return self._avg_avg_cons_extension
    
    def get_avg_runtime(self) -> float:
        return self._avg_runtime
    
    def get_avg_var_sort(self) -> float:
        return self._avg_var_sort
    
    def get_avg_r2_sort(self) -> float:
        return self._avg_r2_sort
    
    def to_bootstrap_plot_dict(self) -> dict: 
        bootstrap_plot_dict = {}
        not_include = ("_causal_inference_tasks", 
                       "_algorithm")
        for attr_name, attr_value in self.__dict__.items():
            if attr_name not in not_include:
                bootstrap_plot_dict[attr_name] = attr_value
        bootstrap_plot_dict["_algorithm"] = self._algorithm.__dict__
        
        return bootstrap_plot_dict

    def _create_causal_inference_tasks(self):

        for counter in range(self._nr_bootstraps):
            bstr_sample = bootstrap_sample(
                datasets=self._data_to_bootstrap_from,
                sample_sizes=self._sample_sizes,
                seed=counter*len(self._sample_sizes)
            )
            self._causal_inference_tasks.append(
                CausalInferenceTask(
                    algorithm=self._algorithm,
                    data=bstr_sample, 
                    true_dag=self._true_dag
                )
            )

    def _run_causal_inference_tasks(self):

        for task in self._causal_inference_tasks:
            task.run_task()

    def _compute_averages(self):

        est_avg_cons_extensions = []
        runtimes = []
        var_sorts = []
        r2_sorts = []
        for task in self._causal_inference_tasks:
            est_avg_cons_extensions.append(task.get_average_cons_extension().values)
            runtimes.append(task.get_runtime())
            var_sorts.append(task.get_var_sort())
            r2_sorts.append(task.get_r2_sort())
        _avg_avg_cons_extension_np = np.average(est_avg_cons_extensions, axis=0)
        self._avg_avg_cons_extension = pd.DataFrame(
            data=_avg_avg_cons_extension_np,
            index=self._bootstrap_variables,
            columns=self._bootstrap_variables
        )
        self._avg_runtime = float(np.average(runtimes, axis=0))
        self._avg_var_sort = float(np.average(var_sorts, axis=0))
        self._avg_r2_sort = float(np.average(r2_sorts, axis=0))

    
class BootstrapComparison:
    
    def __init__(self,
                 comparison_name):
        
        # --- Provided
        self._comparison_name = comparison_name
        # --- Computed later
        self._bootstraps = []
        self._all_var = None
        self._all_var_true_dag = None

    def add_bootstrap(self, bstrp: Bootstrap):

        self._bootstraps.append(bstrp)
        self._check_variable_validity()
        self._update_variables()
        
    def run_comparison(self):
        for bootstrap in self._bootstraps:
            bootstrap.run_bootstrap()

    def get_comparison_plot_dict(self):
        plot_dict = {}
        for bootstrap in self._bootstraps:
            pass

    def to_comparison_plot_dict(self):
        comparison_plot_dict = {}

        for attr_name, attr_value in self.__dict__.items():
            if attr_name == "_bootstraps":
                comparison_plot_dict[attr_name] = [
                    bootstrap.to_bootstrap_plot_dict for bootstrap in self._bootstraps
                ]
            else: 
                comparison_plot_dict[attr_name] = attr_value

        return comparison_plot_dict
    
    def _check_variable_validity(self):
        # Comparison only possible if >= 2 bootstraps have been added
        if len(self._bootstraps) < 2:
            return
            
        # Only addition of variables is allowed 
        if not self._bootstraps[-1].get_bootstrap_variables() == give_superlist(
                first=self._bootstraps[-2].get_bootstrap_variables(),
                second=self._bootstraps[-1].get_bootstrap_variables()):
            raise ValueError("Only addition of variables is allowed from one \
                                bootstrap to the next, not removal")

        # Variables must have the same order
        if not same_order(
                first=self._bootstraps[-2].get_bootstrap_variables(),
                second=self._bootstraps[-1].get_bootstrap_variables()
                ):
            raise ValueError("Variables must have same order from one \
                                bootstrap to the next.")

    def _update_variables(self):          
        # Latest bootstrap must have most variables
        self._all_var = self._bootstraps[-1].get_bootstrap_variables()
        self._all_var_true_dag = self._bootstraps[-1].get_true_dag()
