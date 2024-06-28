
import matplotlib.axes
import networkx as nx

from .edges import Edges
from .nodes import Nodes


class VisGraph:
    """ Draws passed nodes and/or edges onto passed axeses."""

    def __init__(self, 
                 nodes: Nodes,
                 edges: Edges,
                 ax_graph = matplotlib.axes.Axes,
                 ax_legend = matplotlib.axes.Axes,
                 title: str = ""
                 ):
        
        # --- Provided
        self._nodes = nodes
        self._edges = edges
        self._ax_graph = ax_graph
        self._ax_legend = ax_legend
        self._title = title
        
        self._G = nx.DiGraph()

    def draw_edges(self):
        # Unpack 
        edges, colors = self._edges.edgesandcolors
        nx.draw_networkx_edges(G=self._G, 
                                pos=self._nodes.positions, 
                                edgelist=edges, 
                                edge_color=colors,
                                ax=self._ax_graph, 
                                node_size = self._nodes.nodesize)

    def draw_nodes(self):
        # Unpack nodes/colors/visibility/sizes:
        core_var, diff_var, rest_var = self._nodes.var_split
        core_col, diff_col, rest_col = self._nodes.var_cols
        vis, invis = self._nodes.visibility

        common_kwargs = {'G': self._G,
                         'pos': self._nodes.positions,
                         'node_size': self._nodes.nodesize,
                         'edgelist': [],
                         'ax': self._ax_graph,
                         'with_labels': False, 
                         'arrows': True}
        # Core nodes w/o label
        nx.draw_networkx(nodelist=core_var,
                        node_color=core_col,
                        alpha=vis, 
                        **common_kwargs)
        # Diff nodes w/o label
        nx.draw_networkx(nodelist=diff_var,
                        node_color=diff_col,
                        alpha=vis, 
                        **common_kwargs)
        # Rest nodes w/o label and invisible (alpha = 0)
        nx.draw_networkx(nodelist=rest_var,
                        node_color=rest_col,
                        alpha=invis, 
                        **common_kwargs)
        
        # Add labels of core and difference variables
        if self._nodes.latex_transform is None:
            lbs = {label: label for label in core_var+diff_var}
        else:
            lbs = {label: self._nodes.latex_transform(label) for label in core_var+diff_var}

        nx.draw_networkx_labels(G=self._G,
                                pos=self._nodes.positions,
                                labels = lbs,
                                ax = self._ax_graph)


    # Maybe need this later somewhere else 
    def _whiten_axes(self, ax: str):
        if hasattr(self, ax):
            axes = getattr(self, ax)
            if not isinstance(axes, matplotlib.axes.Axes):
                raise TypeError("Alleged axes is not an axes.")
            axes.set_xticks([])
            axes.set_yticks([])
            axes.spines['top'].set_visible(False)
            axes.spines['right'].set_visible(False)
            axes.spines['bottom'].set_visible(False)
            axes.spines['left'].set_visible(False)
            axes.tick_params(axis='both', which='both', length=0)
        else:
            raise AttributeError("Axes not found.")

