"""
Contains the Edge class that provides functionality to compute and draw a list of edges.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.axes

from .helper import AdjGraphs
from .nodes import Nodes
from ..util import is_sub_adj_mat, reduce_to_size, pad_zeros_to_size
from .edgelogic import EdgeLogic
from .edgelogic import TRUE_EDGES, ALL_P, TP, FP, TP_DIFF, FP_DIFF


_EDGE_THRESHOLD = 0.2
_FONTSIZE = 12
_TICK_LOC = "left"
_LABEL_POS = "right"
_NUM_TICKS = 6


class Edges:
    """
    Provides functionality to compute and draw a list of edges.
    """
    def __init__(self, 
                 graphs: AdjGraphs,
                 logic: EdgeLogic,
                 threshold: float = _EDGE_THRESHOLD
                ):
        """
        Passes graphs to be drawn, EdgeLogic to be applied and computes
            a list of edges and edge colors according to these parameters.

        Args:
            graphs (AdjGraphs): The graph(s) to be plotted.
            logic (EdgeLogic): The logic according to which edges and colors
                will be chosen.
            threshold (float, optional): Edges where the EdgeLogic evalutes
                to true will not be chosen if their weight is below this threshold.
                Defaults to _EDGE_THRESHOLD.
        """

        if not logic in (TRUE_EDGES, ALL_P, TP, FP, TP_DIFF, FP_DIFF):
            raise ValueError("Unknown edge logic passed.")

        # Unpack graphs from AdjGraphs object
        ref_graph = graphs.ref_graph
        new_graph = graphs.new_graph
        true_graph = graphs.true_graph

        # Derive graph df which will then be processed according to EdgeLogic 
        if ref_graph is new_graph:
            # Same graph instances passed -> Absolute plot desired
            graph = ref_graph
        elif is_sub_adj_mat(ref_graph, new_graph) & is_sub_adj_mat(new_graph, ref_graph):
            # Same variables, but different values -> Subtract
            graph = new_graph - ref_graph
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
        """
        Draw edges stored in the instance onto the passed axeses.

        Args:
            G (nx.DiGraph): ...
            nodes (Nodes): Contains information about the position 
                and sizes of nodes. Necessary to draw the edges at 
                the correct place.
            ax_graph (matplotlib.axes.Axes): Axes on which the graph
                will be drawn.
            ax_legend (matplotlib.axes.Axes): Axes on which the legend
                (i.e. the used colormap and description) will be drawn.
        """
        # TODO: Create G here locally instead of passing it here.
        nx.draw_networkx_edges(G=G, 
                                pos=nodes.positions, 
                                edgelist=self._edges, 
                                edge_color=self._edge_colors,
                                ax=ax_graph, 
                                node_size = nodes.nodesize)
        
        if self._logic.label is None: # No legend wanted
            return

        # Add colormap legend
        ax_legend.set_aspect(30)
        sm = plt.cm.ScalarMappable(cmap=self._logic.colormap, norm=self._logic.normalizer)
        cbar = plt.colorbar(sm, cax=ax_legend)
        cbar.set_label(self._logic.label, fontsize=_FONTSIZE)
        cbar.ax.yaxis.set_label_position(_LABEL_POS)
        cbar.set_ticks(np.linspace(self._logic.normalizer.vmin, self._logic.normalizer.vmax, num=_NUM_TICKS))
        cbar.ax.yaxis.set_ticks_position(_TICK_LOC)


    def _compute_edges(self):
        """
        Computes list of edges and list edges and edge colors using 
            the attributes passed in the constructor.
        """
        # True graph is larger than graph, hence reduce true graph to size of graph
        true_graph_red = reduce_to_size(self._true_graph, self._graph)
        # Edges that fulfill requirement on the true graph
        true_msk = self._logic.true_graph_comp(true_graph_red, 0) 
        # Edges that fulfill requirements on the graph that shall be plotted
        graph_msk = self._logic.graph_comp(self._graph, self._threshold)
        # Edges that fulfill both requirements and hence will be plotted
        total_msk = true_msk & graph_msk
        # Transform mask into edge list
        self._edges = [(total_msk.index[i], total_msk.columns[j]) for i,j in zip(*np.where(total_msk))]
        # Compute weight list based on edge list
        self._edge_weights = [float(self._graph.at[pos]) for pos in self._edges]
        # Compute color list based on weight list
        self._edge_colors = [self._logic.colormap(self._logic.normalizer(w)) for w in self._edge_weights]




