import numpy as np
import pandas as pd

from .helper import AdjGraphs
from ..util import is_sub_adj_mat, reduce_to_size, pad_zeros_to_size
from .edgelogic import EdgeLogic
from .edgelogic import all_p, tp, fp, tp_diff, fp_diff


EDGE_THRESHOLD = 0.1


class Edges:

    def __init__(self, 
                 graphs: AdjGraphs,
                 threshold: float = EDGE_THRESHOLD
                ):
        
        # Unpack graphs from AdjGraphs object
        ref_graph = graphs.ref_graph
        new_graph = graphs.new_graph
        true_graph = graphs.true_graph

        if is_sub_adj_mat(ref_graph, new_graph) & is_sub_adj_mat(new_graph, ref_graph):
            # Same graphs passed -> Just take first
            if not ref_graph.equals(new_graph):
                raise ValueError(
                    "If both graphs have equal variables, it must be exactly the same")
            graph = ref_graph
        elif is_sub_adj_mat(ref_graph, new_graph):
            # Variable size increases
            ref_graph_pad = pad_zeros_to_size(ref_graph, new_graph)
            graph = new_graph - ref_graph_pad
        elif is_sub_adj_mat(new_graph, ref_graph):
            # Variable size decreses
            new_graph_pad = pad_zeros_to_size(new_graph, ref_graph)
            graph = new_graph_pad - ref_graph
        else:
            raise ValueError("Unclear Error with the passed graphs.")

        # Instantiate self object
        self._graph = graph
        self._true_graph = true_graph
        self._threshold = threshold
        
        # --- computed later
        self._edges = None
        self._edge_weights = None
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
    
    @property
    def edgesandcolors(self):
        assert len(self._edges) == len(self._edge_colors), \
            "Edges and Colors list differ in length."
        return (self._edges, self._edge_colors)
        
    # Public API
    ###

    def _compute(self, logic: EdgeLogic):
        # True graph is larger than graph
        true_graph_red = reduce_to_size(self._true_graph, self._graph)
        true_msk = logic.true_graph_comp(true_graph_red, 0)
        
        graph_msk = logic.graph_comp(self._graph, self._threshold)
        total_msk = true_msk & graph_msk

        self._edges = [(total_msk.index[i], total_msk.columns[j]) for i,j in zip(*np.where(total_msk))]
        self._edge_weights = [float(self._true_graph.at[pos]) for pos in self._edges]
        self._edge_colors = [logic.colormap(logic.normalizer(w)) for w in self._edge_weights]



