import numpy as np
import pandas as pd

from ..util import give_sublist, same_columns
from .edgelogic import EdgeLogic
from .edgelogic import all_p, tp, fp, tp_diff, fp_diff


class Edges:

    def __init__(self, 
                 graph: pd.DataFrame, 
                 true_graph: pd.DataFrame, 
                 threshold: float = 0.1
                ):
        
        self._graph = graph
        self._true_graph = true_graph
        self._threshold = threshold
        
        self._check_entries()
        self._check_consistency()

        # --- computed later
        self._edges = None
        self._edge_colors = None

    ###
    # Public API

    def comp_all(self):
        self._compute(logic=all_p)

    def comp_tp(self):
        self._compute(logic=tp) 

    def comp_fp(self):
        self._compute(logic=fp)

    def comp_tp_diff(self):
        self._compute(logic=tp_diff) 

    def comp_fp_diff(self):
        self._compute(logic=fp_diff) 
    
    def get_edges(self):
        assert len(self._edges) == len(self._edge_colors), \
            "Edges and Colors list differ in length."
        return self._edges
    
    def get_colors(self):
        assert len(self._edges) == len(self._edge_colors), \
            "Edges and Colors list differ in length."
        return self._edge_colors
    
    # Public API
    ###

    def _check_entries(self):
        in_range1 = ((self._graph >= -1) & (self._graph <= 1)).all().all()
        in_range2 = ((self._graph >= 0) & (self._graph <= 1)).all().all()
        if not (in_range1 and in_range2):
            raise ValueError("The passed graphs have incorrect entries.")
    
    def _check_consistency(self):
        bl = (self._graph == give_sublist(self._graph, self._true_graph))
        if not bl:
            raise ValueError("Graph is not a subset of true_graph variable wise.")
        if not same_columns(self._true_graph, self._true_graph.transpose()):
            raise ValueError("True graph has different rows and columns.")
        if not same_columns(self._graph, self._graph.transpose()):
            raise ValueError("Graph has different rows and columns.")

    def _compute(self, logic: EdgeLogic):
        true_msk = logic.true_graph_comp(self._true_graph, 0)
        graph_msk = logic.graph_comp(self._graph, self._threshold)
        common_var = graph_msk.columns # graph variabes are subset of true_graph variables
        total_msk = true_msk.loc[common_var, common_var] & graph_msk.loc[common_var, common_var]
        self._edges = [(total_msk.index[i], total_msk.columns[j]) for i,j in zip(*np.where(total_msk))]
        self._weights = [float(self._true_graph.at[pos]) for pos in self._edges]



