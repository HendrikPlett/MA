import itertools
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import networkx as nx
from typing import Iterable

from .edgelogic import ALL_P, TP, FP, TP_DIFF, FP_DIFF, EdgeLogic
from .helper import AdjGraphs
from .edges import Edges
from .nodes import Nodes
from causalbenchmark.compute import Bootstrap, BootstrapComparison


_FIGSIZE = (10,8)
_AX_WIDTH_RATIO = [7/8, 1/16, 1/16]

_TITLE_FONTSIZE = 16
_TITLE_FONTWEIGHT = "bold"

_TEXT_POS = {'x': 0.03, 'y':0.97}
_TEXT_ALIGN = {'verticalalignment': 'top', 'horizontalalignment': 'left'}
_TEXT_FONTSIZE = 10

_PREC_TITLE = " - Precision"
_TP_CHANGE_TITLE = " - TP change"
_FP_CHANGE_TITLE = " - FP change"


class VisBootstrap:
    def __init__(self, 
                 bstrp: Bootstrap = None, 
                 **kwargs):
        if bstrp is not None:
            graph = bstrp.get_avg_avg_cons_extension()
            true_graph = bstrp.get_true_dag()
            title = bstrp.get_bootstrap_name()
            avg_var_sort = bstrp.get_avg_var_sort()
            avg_r2_sort = bstrp.get_avg_r2_sort()
        else:
            graph = kwargs.get("graph")
            true_graph = kwargs.get("true_graph")
            title = kwargs.get("title")
            avg_var_sort = kwargs.get("avg_var_sort")
            avg_r2_sort = kwargs.get("avg_r2_sort")
        
        # --- Must be assigned in constructor
        self._graph = graph 
        self._true_graph = true_graph
        self._avg_var_sort = avg_var_sort
        self._avg_r2_sort = avg_r2_sort
        self._title = title
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

        # Add title
        _title(
            axes=self._axes[0],
            title=self._title + _PREC_TITLE
        )

        # Add sortability information
        _text_top_left(
            axes=self._axes[0],
            txt=f'Var Sort: {round(self._avg_var_sort,2)} \nR2 Sort: {round(self._avg_r2_sort,2)}'
        )

    def save_fig(self, path: str):
        _save_figure(fig=self._fig, path=path)


class VisBootstrapComparison:

    def __init__(self,
                 bstrp_comp: BootstrapComparison = None,
                 **kwargs
                 ):
        if bstrp_comp is not None:
            bstrps = bstrp_comp.get_bootstraps()
            graphs = [bstrp.get_avg_avg_cons_extension() for bstrp in bstrps]
            titles = [bstrp.get_bootstrap_name() for bstrp in bstrps]
            var_sortabilities = [bstrp.get_avg_var_sort() for bstrp in bstrps]
            r2_sortabilities = [bstrp.get_avg_r2_sort() for bstrp in bstrps]
            true_graph = bstrp_comp.get_all_var_true_DAG()
            nr_bstrps = len(bstrp_comp)
        else:
            pass

        # --- Must be assigned in constructor
        self._graphs = graphs
        self._titles = titles
        self._var_sortabilities = var_sortabilities
        self._r2_sortabilities = r2_sortabilities
        self._true_graph = true_graph
        self._nr_bstrps = nr_bstrps
        self._pos = kwargs.get("pos")
        self._latex_transf = kwargs.get("latex_transf")
        # --- Computed later
        self._fig = None


    def evolution_plot(self, figsize: tuple = _FIGSIZE):
        # Set up figure and axeses in (nr_bootstraps+3, 3) grid
        pass

    def pair_comp_plot(self, figsize: tuple = _FIGSIZE):
        # Set up figure and axeses in (nr_bstrps, nr_bstrps) grid
        self._fig = plt.figure(figsize=(figsize[0]*self._nr_bstrps, 
                               figsize[1]*self._nr_bstrps))
        gs = gridspec.GridSpec(self._nr_bstrps, 3*self._nr_bstrps,
                        width_ratios=[wd/self._nr_bstrps for wd in _AX_WIDTH_RATIO]*self._nr_bstrps, 
                        height_ratios=[1/self._nr_bstrps]*self._nr_bstrps
                        )
        
        for row, col in itertools.product(range(self._nr_bstrps), repeat=2):
            
            # Create axeses
            ax = (self._fig.add_subplot(gs[row, 3*col+0]),
                    self._fig.add_subplot(gs[row, 3*col+1]),
                    self._fig.add_subplot(gs[row, 3*col+2]))
            
            graphs = AdjGraphs(
                    ref_graph=self._graphs[row], # from 
                    new_graph=self._graphs[col], # to
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
                title = self._titles[row]+_PREC_TITLE
                # Add sortability info in precision case only
                _text_top_left(
                    axes=ax[0],
                    txt=f'''Var Sort: {round(self._var_sortabilities[row],2)} \n
                    R2 Sort: {round(self._r2_sortabilities[row],2)}'''
                )
            elif col > row: # TP comparion case 
                _vis(edge_logics=[TP_DIFF], **common_kwargs)
                title = self._titles[row]+" to "+self._titles[col]+_TP_CHANGE_TITLE
            elif col < row: # FP comparison case
                _vis(edge_logics=[FP_DIFF], **common_kwargs) 
                title = self._titles[row]+" to "+self._titles[col]+_FP_CHANGE_TITLE
            else:
                raise ValueError("Comparison with the row/columns not possible.")    
        
            _title(
                axes=ax[0],
                title=title
            )

    def save_fig(self, path: str):
        _save_figure(fig=self._fig, path=path)



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
    
def _title(axes: matplotlib.axes.Axes,
           title: str):
    axes.set_title(label=title,
                   fontsize=_TITLE_FONTSIZE,
                   fontweight = _TITLE_FONTWEIGHT)
    
def _text_top_left(axes: matplotlib.axes.Axes, txt: str):
    axes.text(**_TEXT_POS,
              s=txt, 
              **_TEXT_ALIGN, 
              fontsize = _TEXT_FONTSIZE,
              transform = axes.transAxes)

def _whiten_axes(ax: matplotlib.axes.Axes):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    return ax

def _save_figure(fig: Figure, path: str):
    try:
        fig.savefig(path, format="pdf", dpi=300)
    except Exception as e:
        print(f"Failed to save the figure: {e}")
