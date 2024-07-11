"""
Class 'Bootstrap' to create CausalInferenceTasks with bootstrapped
    datasets.
Class 'BootstrapComparison' to compare Bootstrap instances.
"""

# Standard 
from typing import Iterable
import numpy as np
import pandas as pd

# Third party

# Own
from .algorithms import Algorithm
from ..util import same_columns, bootstrap_sample, give_superlist, same_order, variables_increase
from .causal_inference_task import CausalInferenceTask

class Bootstrap():
    """
    Create specified number of bootstrap samples from passed data, 
        use theses samples to create CausalInferenceTasks, fit them
        and compute average values across all bootstrap samples.
    """
    def __init__(self, 
                 bootstrap_name: str, 
                 algorithm: Algorithm,
                 data_to_bootstrap_from: Iterable[pd.DataFrame], 
                 sample_sizes: tuple, 
                 nr_bootstraps: int,
                 true_dag: pd.DataFrame):
        """
        Initialize Bootstrap, passed variables cannot be changed later on.

        Args:
            bootstrap_name (str): Ideally describing the passed arguments.
            algorithm (Algorithm): The Algorithm used to fit a structure 
                to the data.
            data_to_bootstrap_from (Iterable[pd.DataFrame]): All passed 
                dfs must have the same columns, will also be checked.
            sample_sizes (tuple): Either a percentage or number. One size
                for each passed df.
            nr_bootstraps (int): How many bootstrap samples (and thus 
                CausalInferenceTask instances) will be created.
            true_dag (pd.DataFrame): The causal structure according to
                which the data_to_bootstrap_from was generated. Must include
                the same variables as the passed data.
        """
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
        self._causal_inference_tasks = []
        self._avg_avg_cons_extension = None
        self._avg_runtime = None
        self._avg_var_sort = None
        self._avg_r2_sort = None

    def run_bootstrap(self):
        """
        Creates, runs and processes CausalInferceTask instances
            according to arguments passed in the constructor.
        """
        self._create_causal_inference_tasks()
        self._run_causal_inference_tasks()
        self._compute_averages()

    def get_bootstrap_name(self) -> str:
        """Get the name as passed in constructor."""
        return self._bootstrap_name
    
    def get_bootstrap_variables(self) -> list:
        """Get the variables passed via the dfs and true_dag in constructor."""
        return self._bootstrap_variables

    def get_true_dag(self) -> pd.DataFrame:
        """Get the true DAG as passed in constructor."""
        return self._true_dag
    
    def get_avg_avg_cons_extension(self) -> pd.DataFrame:
        """
        Gets graph that is:
            -averaged over all consistent extensions within a CI-Task
            -averaged this average across all bootstrapped CI-Tasks.
        """
        return self._avg_avg_cons_extension
    
    def get_avg_runtime(self) -> float:
        """Average runtime of 'Algorithm' averaged over all bootstrapped CI-Tasks."""
        return self._avg_runtime
    
    def get_avg_var_sort(self) -> float:
        """
        Average Var-Sortability of the bootstrapped dataset across 
            all bootstrapped datasets.
        """
        return self._avg_var_sort
    
    def get_avg_r2_sort(self) -> float:
        """
        Average R2-Sortability of the bootstrapped dataset across 
            all bootstrapped datasets.
        """
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
        """
        Creates nr_bootstraps datasets and uses them
            to create CausalInferenceTask instances.
        """
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
        """Iterate over each CausalInferenceTasks instance and apply run function."""
        for task in self._causal_inference_tasks:
            task.run_task()

    def _compute_averages(self):
        """
        After each CausalInferenceTask instance is fitted, compute
            - average runtime of 'Algorithm' across all tasks,
            - average sortability of the boostrapped datasets across all tasks,
            - average average consistent extension of fitted structure across all tasks.
        """
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
    """
    Class comparing several Bootstrap instances. 
    In particular tracking differences in the variables used.
    """
    def __init__(self,
                 comparison_name: str):
        """_summary_

        Args:
            comparison_name (str): Ideally the factor by which 
                the passed Bootstrap instances differ.
        """
        # --- Provided
        self._comparison_name = comparison_name
        # --- Computed later
        self._bootstraps = []
        self._all_var = None
        self._all_var_true_dag = None

    def add_bootstrap(self, bstrp: Bootstrap):
        """
        Store the passed Bootstrap instance if the used
            variables in that instance are valid, i.e. if
            they are a superset of the variables used in
            previously passed Bootstrap instances.
        """
        self._check_variable_validity(bstrp)
        self._bootstraps.append(bstrp)
        self._update_variables()
        
    def run_comparison(self):
        """Run each passed Bootstrap instance."""
        for bootstrap in self._bootstraps:
            bootstrap.run_bootstrap()

    def get_bootstraps(self) -> list[Bootstrap]:
        """Get the list containing all passed Bootstrap instances."""
        return self._bootstraps
    
    def get_all_var_true_DAG(self) -> pd.DataFrame:
        """Get the true DAG that contains all variables that are used 
            in at least one passed Bootsrap instance."""
        return self._all_var_true_dag

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
    
    def _check_variable_validity(self, bstrp: Bootstrap):
        """
        Compares passed Bootstrap instance to previously passed Bootstrap instance.
        Raises ValueError if:
            - The variables of the new Bootstrap are not a superset of the ones 
                of the old Bootstrap.
            - The common variables of the two Bootstraps are in different order.
        """
        # Comparison only possible if >= 1 bootstraps have been added
        if len(self._bootstraps) < 1:
            return
            
        # Only addition of variables is allowed 
        if not variables_increase(self._bootstraps[-1].get_bootstrap_variables(), 
                                  bstrp.get_bootstrap_variables()):
            raise ValueError("Only addition of variables is allowed from one \
                                bootstrap to the next, not removal")

        # Variables must have the same order
        if not same_order(
                first=self._bootstraps[-1].get_bootstrap_variables(),
                second=bstrp.get_bootstrap_variables()
                ):
            raise ValueError("Variables must have same order from one \
                                bootstrap to the next.")

    def _update_variables(self):          
        """
        Save variables and true DAG of latest Bootstrap instance, 
            as - by requirements - has the most variables.
        """
        self._all_var = self._bootstraps[-1].get_bootstrap_variables()
        self._all_var_true_dag = self._bootstraps[-1].get_true_dag()

    def __len__(self):
        """Number of passed Bootstrap instances."""
        return len(self._bootstraps)
    
    def __getitem__(self, index):
        """Get bootstrap instance at passed index."""
        return self._bootstraps[index]
        
