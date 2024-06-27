import pandas as pd

from ..util import enforce_valid_adj_mat, variables_increase, enforce_no_duplicates

class Nodes:

    def __init__(self, 
                 ref_graph: pd.DataFrame, 
                 new_graph: pd.DataFrame, 
                 all_var: list[str]
                 ):
        
        # Validity of input
        enforce_valid_adj_mat(ref_graph)
        enforce_valid_adj_mat(new_graph)
        enforce_no_duplicates(all_var)
        if not variables_increase(ref_graph.index.to_list(), all_var):
            raise ValueError("RefGraph has too many variables.")
        if not variables_increase(new_graph.index.to_list(), all_var):
            raise ValueError("NewGraph has too many variables.")

        self._all_var = all_var
        # Check which passed graph is smaller
        if variables_increase(ref_graph.index.to_list(), new_graph.index.to_list()):
            self._smaller_graph = ref_graph
            self._larger_graph = new_graph
        elif variables_increase(new_graph.index.to_list(), ref_graph.index.to_list()):
            self._smaller_graph = new_graph
            self._larger_graph = ref_graph
        else:
            ValueError("Unclear Error with the passed graphs.")

        # --- computed later
        self._core_var = []
        self._diff_var = []
        self._rest_var = []

    @property
    def var_split(self):
        return (self._core_var, self._diff_var, self._rest_var)

    def compute_var_groups(self):
        self._core_var = self._smaller_graph.columns.to_list()
        self._diff_var = list(set(self._larger_graph).difference(set(self._smaller_graph)))
        self._rest_var = list(set(self._all_var).difference(set(self._larger_graph)))
