import pandas as pd

from .helper import AdjGraphs
from ..util import is_sub_adj_mat

_CORE_COL = 'lightblue'
_DIFF_COL_ADD = 'deepskyblue'
_DIFF_COL_DEL = 'aliceblue'
_REST_COL = 'white'

_NODESIZE = 400
_VISIBLE = 1.0
_INVISIBLE = 0.0

class Nodes:

    def __init__(self,
                 graphs: AdjGraphs, 
                 pos: None,
                 latex_transf: None
                 ):
        
        # Unpack graphs from AdjGraphs object
        ref_graph = graphs.ref_graph
        new_graph = graphs.new_graph
        true_graph = graphs.true_graph

        # Check which passed graph is smaller
        if is_sub_adj_mat(ref_graph, new_graph):
            smaller_graph_var = ref_graph.index.to_list()
            larger_graph_var = new_graph.index.to_list()
            diff_col = _DIFF_COL_ADD 
        elif is_sub_adj_mat(new_graph, ref_graph):
            smaller_graph_var = new_graph.index.to_list()
            larger_graph_var = ref_graph.index.to_list()
            diff_col = _DIFF_COL_DEL 
        else:
            ValueError("Unclear Error with the passed graphs.")
        
        # --- Instantiate self object
        # Variables
        self._smaller_graph_var = smaller_graph_var
        self._larger_graph_var = larger_graph_var
        self._all_var = true_graph.index.to_list()
        # Colors
        self._core_col = _CORE_COL
        self._diff_col = diff_col
        self._rest_col = _REST_COL
        # Positions
        self._pos = pos
        self._latex_transf = latex_transf
        # Sizes
        self._node_size = _NODESIZE
        self._visible = _VISIBLE
        self._invisible = _INVISIBLE
        
        # --- computed later
        self._core_var = []
        self._diff_var = []
        self._rest_var = []

    @property
    def var_split(self):
        return (self._core_var, self._diff_var, self._rest_var)

    @property
    def var_cols(self):
        return (self._core_col, self._diff_col, self._rest_col)
    
    @property
    def positions(self):
        return self._pos
    
    @property
    def latex_transform(self):
        return self._latex_transf
    
    @property
    def nodesize(self):
        return self._node_size
    
    @property
    def visibility(self):
        return (self._visible, self._invisible) 

    def compute_var_groups(self):
        self._core_var = self._smaller_graph_var
        self._diff_var = list(set(self._larger_graph_var).difference(self._smaller_graph_var))
        self._rest_var = list(set(self._all_var).difference(self._larger_graph_var))