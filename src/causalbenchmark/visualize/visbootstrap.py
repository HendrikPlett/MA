import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

from .edgelogic import ALL_P, TP, FP, TP_DIFF, FP_DIFF
from .helper import AdjGraphs
from .edges import Edges
from .nodes import Nodes
from causalbenchmark.compute import Bootstrap, BootstrapComparison


_FIGSIZE = (10,8)
_AX_WIDTH_RATIO = [7/8, 1/16, 1/16]

class VisBootstrap:
    def __init__(self, 
                 bstrp: Bootstrap = None, 
                 **kwargs):
        if bstrp is None:
            self._graph = kwargs['graph']
            self._true_graph = kwargs['true_graph']
        else:
            self._graph = bstrp.get_avg_avg_cons_extension()
            self._true_graph = bstrp.get_true_dag()
        
        # --- Computed later
        self._G = None
        self._fig = None
        self._ax1 = None
        self._ax2 = None
        self._ax3 = None
    
    def vis_precision(self, figsize: tuple=_FIGSIZE):
        
        # Set up figure and axes in (1,3) grid with desired width ratios
        self._fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 3, width_ratios=_AX_WIDTH_RATIO)
        self._ax1 = self._fig.add_subplot(gs[0, 0])
        self._ax2 = self._fig.add_subplot(gs[0, 1])
        self._ax3 = self._fig.add_subplot(gs[0, 2])

        # Pool graphs
        graphs = AdjGraphs(ref_graph=self._graph,
                           new_graph=self._graph,
                           true_graph=self._true_graph)
        
        # Create empty graph
        self._G = nx.DiGraph()

        # Add nodes
        nodes = Nodes(graphs=graphs)
        nodes.draw_nodes(G=self._G, ax_graph=self._ax1)

        # Add TP and FP edges
        edges_tp = Edges(graphs=graphs, logic=TP)
        edges_tp.draw_edges(G=self._G,
                            nodes=nodes,
                            ax_graph=self._ax1,
                            ax_legend=self._ax2)
        edges_fp = Edges(graphs=graphs, logic=FP)
        edges_fp.draw_edges(G=self._G,
                            nodes=nodes,
                            ax_graph=self._ax1,
                            ax_legend=self._ax3)


class VisBootstrapComparison:


    def __init__(self):
        pass

    def evolution_plot(self):
        pass

    def pair_comp_plot(self):
        pass



def _whiten_axes(ax: matplotlib.axes.Axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    return ax
