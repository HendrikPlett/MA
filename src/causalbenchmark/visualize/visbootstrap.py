import itertools
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from typing import Iterable

from .edgelogic import ALL_P, TP, FP, TP_DIFF, FP_DIFF, EdgeLogic
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
        if bstrp is not None:
            graph = bstrp.get_avg_avg_cons_extension()
            true_graph = bstrp.get_true_dag()
        else:
            graph = kwargs.get("graph")
            true_graph = kwargs.get("true_graph")
        
        # --- Must be assigned in constructor
        self._graph = graph 
        self._true_graph = true_graph
        self._pos = kwargs.get("pos")
        self._latex_transf = kwargs.get("latex_transf")
        # --- Computed later
        self._G = None
        self._fig = None
        self._axes = None
    
    def vis_precision(self, figsize: tuple=_FIGSIZE):
        
        # Set up figure and axeses in (1,3) grid with desired width ratios
        self._fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 3, width_ratios=_AX_WIDTH_RATIO)
        self._axes = [self._fig.add_subplot(gs[0, 0]),
                self._fig.add_subplot(gs[0, 1]),
                self._fig.add_subplot(gs[0, 2])]

        # Pool graphs
        graphs = AdjGraphs(ref_graph=self._graph,
                           new_graph=self._graph,
                           true_graph=self._true_graph)
        
        # Plot graph onto created axeses
        _vis(
            axes=self._axes,
            graphs=graphs,
            pos=self._pos,
            latex_transf=self._latex_transf,
            edge_logics=[TP, FP] # TP and FP for precision
        )


class VisBootstrapComparison:

    def __init__(self,
                 bstrp_comp: BootstrapComparison = None,
                 **kwargs
                 ):
        if bstrp_comp is not None:
            graphs = [bstrp.get_avg_avg_cons_extension() for bstrp in bstrp_comp.get_bootstraps()]
            true_graph = bstrp_comp.get_all_var_true_DAG()
            nr_bstrps = len(bstrp_comp)
        else:
            pass

        # --- Must be assigned in constructor
        self._graphs = graphs
        self._true_graph = true_graph
        self._nr_bstrps = nr_bstrps
        self._pos = kwargs.get("pos")
        self._latex_transf = kwargs.get("latex_transf")

    def evolution_plot(self, figsize: tuple = _FIGSIZE):
        pass

    def pair_comp_plot(self):
        # Set up figure and axeses in (nr_bstrps, nr_bstrps) grid
        fig = plt.figure(figsize=(_FIGSIZE[0]*self._nr_bstrps, 
                             _FIGSIZE[1]*self._nr_bstrps))
        gs = gridspec.GridSpec(self._nr_bstrps, 3*self._nr_bstrps,
                        width_ratios=[wd/self._nr_bstrps for wd in _AX_WIDTH_RATIO]*self._nr_bstrps, 
                        height_ratios=[1/self._nr_bstrps]*self._nr_bstrps
                        )
        
        for row, col in itertools.product(range(self._nr_bstrps), repeat=2):
            
            # Create axeses
            ax = (fig.add_subplot(gs[row, 3*col+0]),
                    fig.add_subplot(gs[row, 3*col+1]),
                    fig.add_subplot(gs[row, 3*col+2]))
            
            graphs = AdjGraphs(
                    ref_graph=self._graphs[row],
                    new_graph=self._graphs[col],
                    true_graph=self._true_graph
                )

            common_kwargs = {
                'axes': ax,
                'graphs': graphs,
                'pos': self._pos,
                'latex_transf': self._latex_transf
            }

            if row == col: # Precision case 
                _vis(edge_logics=[TP, FP], **common_kwargs)
            elif col > row: # TP comparion case 
                _vis(edge_logics=[TP_DIFF], **common_kwargs)
            elif col < row: # FP comparison case
                _vis(edge_logics=[FP_DIFF], **common_kwargs)               


def _vis(axes: Iterable[matplotlib.axes.Axes],
         graphs: AdjGraphs,
         pos: dict,
         latex_transf,
         edge_logics: list[EdgeLogic]):
    
        # Create empty graph
        G = nx.DiGraph()

        # Add nodes
        nodes = Nodes(graphs=graphs,
                      pos=pos,
                      latex_transf=latex_transf)
        nodes.draw_nodes(G=G, ax_graph=axes[0])

        # Add desired edges
        for logic, ax in zip(edge_logics, axes[1:]):
            edges = Edges(graphs=graphs, logic=logic)
            edges.draw_edges(G=G,
                             nodes=nodes,
                             ax_graph=axes[0],
                             ax_legend=ax)

        # Whiten non-used axeses            
        for ax in axes[len(edge_logics)+1:]:
            _whiten_axes(ax)
    

def _whiten_axes(ax: matplotlib.axes.Axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    return ax