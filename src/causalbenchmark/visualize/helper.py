import pandas as pd

from ..util import (enforce_valid_adj_mat,
                    enforce_binary_adj_mat, 
                    enforce_sub_adj_mat)


class AdjGraphs:
    """Store graphs in a structure that is helpful in plotting."""
    def __init__(self, 
                 ref_graph: pd.DataFrame, 
                 new_graph: pd.DataFrame, 
                 true_graph: pd.DataFrame
                 ):
        """
        Initialize object and ensure validity for plotting.

        Args:
            ref_graph (pd.DataFrame): The graph that will be plotted
                in an absolute plot or the reference graph in a comparison
                plot.
            new_graph (pd.DataFrame): The new graph in a comparison plot, 
                must have a superset of variables compared to the ref_graph.
            true_graph (pd.DataFrame): The true graph, must have a superset
                of variables compared to the other two graphs.
        """
        # Enforce validity of all passed graphs
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
        """Get the valid reference graph."""
        return self._ref_graph
    
    @property
    def new_graph(self) -> pd.DataFrame:
        """Get the valid new graph."""
        return self._new_graph

    @property
    def true_graph(self) -> pd.DataFrame:
        """Get the valid true graph."""
        return self._true_graph

    
