import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.axes

from .helper import AdjGraphs
from .nodes import Nodes
from ..util import is_sub_adj_mat, reduce_to_size, pad_zeros_to_size
from .edgelogic import EdgeLogic
from .edgelogic import ALL_P, TP, FP, TP_DIFF, FP_DIFF


_EDGE_THRESHOLD = 0.1
_FONTSIZE = 12
_TICK_LOC = "left"
_LABEL_POS = "right"


class Edges:

    def __init__(self, 
                 graphs: AdjGraphs,
                 logic: EdgeLogic,
                 threshold: float = _EDGE_THRESHOLD
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

        # --- Instantiate self object
        self._graph = graph
        self._true_graph = true_graph
        self._logic = logic
        self._threshold = threshold
        
        # --- Computed later
        self._edges = None
        self._edge_weights = None
        self._edge_colors = None

        # --- Compute
        self._compute_edges()


    def draw_edges(self, 
                   G: nx.DiGraph,
                   nodes: Nodes,
                   ax_graph: matplotlib.axes.Axes,
                   ax_legend: matplotlib.axes.Axes):
        
        nx.draw_networkx_edges(G=G, 
                                pos=nodes.positions, 
                                edgelist=self._edges, 
                                edge_color=self._edge_colors,
                                ax=ax_graph, 
                                node_size = nodes.nodesize)
        
        ax_legend.set_aspect(30)
        sm = plt.cm.ScalarMappable(cmap=self._logic.colormap, norm=self._logic.normalizer)
        cbar = plt.colorbar(sm, cax=ax_legend, ticklocation = _TICK_LOC)
        cbar.set_label(self._logic.label, fontsize=_FONTSIZE)
        cbar.ax.yaxis.set_label_position(_LABEL_POS)


    def _compute_edges(self):
        # True graph is larger than graph
        true_graph_red = reduce_to_size(self._true_graph, self._graph)
        true_msk = self._logic.true_graph_comp(true_graph_red, 0)
        
        graph_msk = self._logic.graph_comp(self._graph, self._threshold)
        total_msk = true_msk & graph_msk

        self._edges = [(total_msk.index[i], total_msk.columns[j]) for i,j in zip(*np.where(total_msk))]
        self._edge_weights = [float(self._true_graph.at[pos]) for pos in self._edges]
        self._edge_colors = [self._logic.colormap(self._logic.normalizer(w)) for w in self._edge_weights]




