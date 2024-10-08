"""
Contains the Node class that provides functionality to compute and draw a list of edges.
"""

from typing import Callable
import pandas as pd
import networkx as nx
import matplotlib.axes

from .helper import AdjGraphs
from ..util import is_sub_adj_mat

# Node colors
_CORE_COL = 'lightblue'
_DIFF_COL_ADD = 'deepskyblue'
_DIFF_COL_DEL = 'aliceblue'
_REST_COL = 'white'

# Node size and visibility
_NODESIZE = 400
_VISIBLE = 1.0
_INVISIBLE = 0.0

class Nodes:
    """Provides functionality to compute and draw a set of nodes."""
    def __init__(self,
                 graphs: AdjGraphs, 
                 pos: dict = None,
                 latex_transf: Callable[[str], str] = None
                 ):
        """
        Instantiates instance and computes the node and color sets
            based on the input.

        Args:
            graphs (AdjGraphs): The graphs to be plotted.
            pos (dict, optional): Where nodes shall be plotted.
                Key: Node name, Value: (float, float) position on the grid.
                Defaults to None.
            latex_transf (Callable[[str], str], optional): Function turning the 
                variable strings into a latex representation that will be used
                for the plot. Defaults to None.
        """
        
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
        
        # --- Computed later
        self._core_var = []
        self._diff_var = []
        self._rest_var = []

        # --- Compute
        self._compute_var_groups()


    def draw_nodes(self, G: nx.DiGraph,
                   ax_graph: matplotlib.axes.Axes):
        """
        Draws computed nodes and colors onto the passed axes.

        Args:
            G (nx.DiGraph): ...
            ax_graph (matplotlib.axes.Axes): The axes to draw
                the nodes on.
        """

        # Use circular layout of no positions are passed
        if self._pos is None:
            self._pos = nx.circular_layout([*self._core_var,
                                            *self._diff_var,
                                            *self._rest_var])

        common_kwargs = {'G': G,
                         'pos': self._pos,
                         'node_size': _NODESIZE,
                         'edgelist': [],
                         'ax': ax_graph,
                         'with_labels': False, 
                         'arrows': True}
        # Core nodes w/o label
        nx.draw_networkx(nodelist=self._core_var,
                        node_color=self._core_col,
                        alpha=_VISIBLE, 
                        **common_kwargs)
        # Diff nodes w/o label
        nx.draw_networkx(nodelist=self._diff_var,
                        node_color=self._diff_col,
                        alpha=_VISIBLE, 
                        **common_kwargs)
        # Rest nodes w/o label and invisible (alpha = 0)
        nx.draw_networkx(nodelist=self._rest_var,
                        node_color=self._rest_col,
                        alpha=_INVISIBLE, 
                        **common_kwargs)
        
        # Add labels of core and difference variables
        if self._latex_transf is None:
            lbs = {label: label for label in self._core_var+self._diff_var}
        else:
            lbs = {label: self._latex_transf(label) for label in self._core_var+self._diff_var}

        nx.draw_networkx_labels(G=G,
                                pos=self._pos,
                                labels = lbs,
                                ax = ax_graph)

    @property
    def positions(self):
        """Get the positions used for plotting."""
        return self._pos
    
    @property
    def nodesize(self):
        """Get the node size used for plotting."""
        return self._node_size    

    def _compute_var_groups(self):
        """Compute three groups of variables."""
        self._core_var = self._smaller_graph_var
        self._diff_var = list(set(self._larger_graph_var).difference(self._smaller_graph_var))
        self._rest_var = list(set(self._all_var).difference(self._larger_graph_var))