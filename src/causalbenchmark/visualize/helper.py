import pandas as pd

from ..util import (enforce_valid_adj_mat,
                    enforce_binary_adj_mat, 
                    enforce_sub_adj_mat)


class AdjGraphs:

    def __init__(self, 
                 ref_graph: pd.DataFrame, 
                 new_graph: pd.DataFrame, 
                 true_graph: pd.DataFrame
                 ):
        
        enforce_valid_adj_mat(ref_graph)
        enforce_valid_adj_mat(new_graph)
        enforce_binary_adj_mat(true_graph)
        enforce_sub_adj_mat(ref_graph, true_graph)
        enforce_sub_adj_mat(new_graph, true_graph)
        
        self._ref_graph = ref_graph
        self._new_graph = new_graph
        self._true_graph = true_graph

    @property
    def ref_graph(self) -> pd.DataFrame:
        return self._ref_graph
    
    @property
    def new_graph(self) -> pd.DataFrame:
        return self._new_graph

    @property
    def true_graph(self) -> pd.DataFrame:
        return self._true_graph

    
