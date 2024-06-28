import numpy as np
import pandas as pd

from .helper import AdjGraphs
from ..util import is_sub_adj_mat, reduce_to_size, pad_zeros_to_size
from .edgelogic import EdgeLogic
from .edgelogic import ALL_P, TP, FP, TP_DIFF, FP_DIFF


EDGE_THRESHOLD = 0.1


class Edges:

    def __init__(self, 
                 graphs: AdjGraphs,
                 logic: EdgeLogic,
                 threshold: float = EDGE_THRESHOLD
                ):
        
        if not logic in (ALL_P, TP, FP, TP_DIFF, FP_DIFF):
            raise ValueError("Unknown edge logic passed.")

        # Unpack graphs from AdjGraphs object
        ref_graph = graphs.ref_graph
        new_graph = graphs.new_graph
        true_graph = graphs.true_graph

        # Handle graph object
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
        self._logic = logic
        self._threshold = threshold
        
        # --- computed later
        self._edges = None
        self._edge_weights = None
        self._edge_colors = None

        self._colormap = None
        self._normalizer = None


    ###
    # Public API

    def comp_edges(self):
        self._compute_edges()
    
    @property
    def edgesandcolors(self):
        assert len(self._edges) == len(self._edge_colors), \
            "Edges and Colors list differ in length."
        return (self._edges, self._edge_colors)
    
    @property
    def usedlogic(self):
        return self._logic
    
    # Public API
    ###

    def _compute_edges(self):
        # True graph is larger than graph
        true_graph_red = reduce_to_size(self._true_graph, self._graph)
        true_msk = self._logic.true_graph_comp(true_graph_red, 0)
        
        graph_msk = self._logic.graph_comp(self._graph, self._threshold)
        total_msk = true_msk & graph_msk

        self._edges = [(total_msk.index[i], total_msk.columns[j]) for i,j in zip(*np.where(total_msk))]
        self._edge_weights = [float(self._true_graph.at[pos]) for pos in self._edges]
        self._edge_colors = [self._logic.colormap(self._logic.normalizer(w)) for w in self._edge_weights]




