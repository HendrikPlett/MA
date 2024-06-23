# Standard 
import pandas as pd
from typing import Iterable

# Third party

# Own
from algorithms import Algorithm
from bootstrap import Bootstrap
from util import give_superlist, same_order


class BootstrapComparison:
    
    def __init__(self,
                 comparison_name):
        
        # --- Provided
        self._comparison_name = comparison_name
        # --- Computed later
        self._bootstraps = []
        self._all_var = None
        self._all_var_true_dag = None

    def add_bootstrap(self, 
                    bootstrap_name: str, 
                    algorithm: Algorithm,
                    data_to_bootstrap_from: Iterable[pd.DataFrame], 
                    sample_sizes: tuple, 
                    nr_bootstraps: int,
                    true_dag: pd.DataFrame):
        
        # Create and add Bootstrap instance
        self._bootstraps.append(
                Bootstrap(
                    bootstrap_name=bootstrap_name, 
                    algorithm=algorithm,
                    data_to_bootstrap_from=data_to_bootstrap_from, 
                    sample_sizes=sample_sizes,
                    nr_bootstraps=nr_bootstraps,
                    true_dag=true_dag
                )
            )
        
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

    def _update_variables(self):
        
        # Comparison only possible if two bootstraps have been added
        if len(self._bootstraps) >= 2:
            
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
                
        # Update variables (latest bootstrap must have most variables)
        self._all_var = self._bootstraps[-1].get_bootstrap_variables()
            
        # Update true DAG
        self._all_var_true_dag = self._bootstraps[-1].get_true_dag()
