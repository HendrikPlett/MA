"""
Module containing the main functionality to visualize Bootstrap 
    and BootstrapComparison objects.
"""

# Standard 
from itertools import product
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
import networkx as nx
from typing import Iterable
from datetime import timedelta

# Own
from .edgelogic import TRUE_EDGES, TP, FP, TP_DIFF, FP_DIFF, EdgeLogic
from .helper import AdjGraphs
from .edges import Edges
from .nodes import Nodes
from causalbenchmark.compute import Bootstrap, BootstrapComparison

matplotlib.use('Agg')

_FIGSIZE = (10,8) # Sizes of each subfigure
_AX_WIDTH_RATIO = [7/8, 1/16, 1/16] # Ratio of the three axeses in a subfigure
_EV_COL = [0,1,2] # [Main axes for the graph, First legend axes, Second legend axes]

# --- Parameters for the title that appears above each subfigure
_TITLE_FONTSIZE = 16
_TITLE_FONTWEIGHT = "bold"
_TRUE_TITLE = "True DAG"
# Title ending based on which plot is drawn
_PREC_TITLE = " - Precision"
_TP_TITLE = " - True Positive"
_FP_TITLE = " - False Positive"
_TP_CHANGE_TITLE = " - TP change"
_FP_CHANGE_TITLE = " - FP change"

# --- Parameters for the text that is put into the graph axes
_TEXT_POS = {'x': 0.03, 'y':0.97}
_TEXT_ALIGN = {'verticalalignment': 'top', 'horizontalalignment': 'left'}
_TEXT_FONTSIZE = 10

# --- Parameters for edge default values
_EDGE_THRESHOLD_ABS = 0.2
_EDGE_THRESHOLD_COMP = 0.05


class VisBootstrap:
    """Functionality to visualize a single bootstrap."""
    def __init__(self, 
                 bstrp: Bootstrap = None, 
                 **kwargs):
        """TODO"""
        if isinstance(bstrp, Bootstrap):
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
        """TODO: Maybe make for flexible and pass edge_logic
            as an argument"""
        self._setup_precision_fig_and_ax(figsize)
    
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

    def _setup_precision_fig_and_ax(self, figsize: tuple):
        """Setup (1,3) axes grid in a figure and store them as instance variables."""
        self._fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 3, width_ratios=_AX_WIDTH_RATIO)
        self._axes = [self._fig.add_subplot(gs[0, 0]),
                self._fig.add_subplot(gs[0, 1]),
                self._fig.add_subplot(gs[0, 2])]


class VisBootstrapComparison:
    """Functionality to visualize a multiple bootstraps."""
    def __init__(self,
                 bstrp_comp: BootstrapComparison = None,
                 **kwargs
                 ):
        """TODO: Define constructor via dict also."""
        if isinstance(bstrp_comp, BootstrapComparison):
            bstrps = bstrp_comp.get_bootstraps()
            graphs = [bstrp.get_avg_avg_cons_extension() for bstrp in bstrps]
            titles = [bstrp.get_bootstrap_name() for bstrp in bstrps]
            var_sortabilities = [round(bstrp.get_avg_var_sort(),2) for bstrp in bstrps]
            r2_sortabilities = [round(bstrp.get_avg_r2_sort(),2) for bstrp in bstrps]
            runtimes = [round(bstrp.get_avg_runtime()) for bstrp in bstrps]
            no_cons_extensions = [round(bstrp.get_avg_no_cons_extension(),2) for bstrp in bstrps]
            alg_crashed = [round(bstrp.get_avg_alg_crashed(),2) for bstrp in bstrps]
            true_graph = bstrp_comp.get_all_var_true_DAG()
            nr_bstrps = len(bstrp_comp)
        else:
            pass

        # --- Must be assigned in constructor
        self._graphs = graphs
        self._titles = titles
        self._var_sortabilities = var_sortabilities
        self._r2_sortabilities = r2_sortabilities
        self._runtimes = runtimes
        self._no_cons_extensions = no_cons_extensions
        self._alg_crashed = alg_crashed
        self._true_graph = true_graph
        self._nr_bstrps = nr_bstrps
        self._pos = kwargs.get("pos")
        self._latex_transf = kwargs.get("latex_transf")
        # --- Computed later
        self._fig = None
        self._gs = None


    def evolution_plot(self, figsize: tuple = _FIGSIZE):
        """Each bootstrap in the list of bootstraps is compared to its predecessor."""
        # Set up figure and axes grid and save as instance variables
        self._setup_evolution_fig_and_grid(figsize)

        # Iterate over the axes grid
        for row, col in product(range(-1, self._nr_bstrps+1), _EV_COL):
            # Get the three axeses required for a subfigure
            ax = self._add_axes(row+1, col)
            
            # Get the desired plot-type for that row/col combination
            res = self._evolution_logic_graphs_title(ax, row, col)
            if res is False: # No plot needed, only axeses are whitened
                continue
            else:
                edge_logics, ref_graph, new_graph, title = res # Unpack
                        
            # Add nodes, edges and legend to the axeses
            graphs = AdjGraphs(
                    ref_graph=ref_graph, # from 
                    new_graph=new_graph, # to
                    true_graph=self._true_graph)
            _vis(
                axes=ax,
                graphs=graphs,
                pos=self._pos,
                latex_transf=self._latex_transf,
                edge_logics=edge_logics
            )
            # Add sortability information if needed for this row/col combination
            self._evolutions_add_sortability(ax[0], row, col)
            # Add title
            _title(axes=ax[0],title=title)

    def pair_comp_plot(self, figsize: tuple = _FIGSIZE):
        """Each bootstrap in the list of bootstraps is compared to all other bootstraps."""
        # Set up figure and axes grid and save as instance variables
        self._setup_pairwise_fig_and_grid(figsize)

        # Iterate over axes grid
        for row, col in product(range(self._nr_bstrps), repeat=2):

            # Get the three axeses required for a subfigure
            ax = self._add_axes(row, col)

            # Get the desired plot-type for that row/col combination
            edge_logics, title = self._pairwise_logic_and_title(row, col)

            # Add nodes, edges and legend to the axeses
            graphs = AdjGraphs(
                ref_graph=self._graphs[row], # from 
                new_graph=self._graphs[col], # to
                true_graph=self._true_graph
            )
            _vis(
                axes=ax,
                graphs=graphs,
                pos=self._pos,
                latex_transf=self._latex_transf,
                edge_logics=edge_logics
            )
            # Add sortability info if needed
            self._pairwise_add_sortability(ax[0], row, col)
            # Add title
            _title(axes=ax[0],title=title)

    def save_fig(self, path: str):
        """Save self._fig under the passed path."""
        _save_figure(fig=self._fig, path=path)

    def _add_axes(self, row: int, col: int):
        """Adds and saves three axeses in the row/col position in the gridspace."""
        ax = (self._fig.add_subplot(self._gs[row, 3*col+0]),
                    self._fig.add_subplot(self._gs[row, 3*col+1]),
                    self._fig.add_subplot(self._gs[row, 3*col+2]))
        return ax
    
    #------------------------------------------------------
    # Private fcts for pairwise comparison

    def _setup_pairwise_fig_and_grid(self, figsize: tuple):
        """Initializes the figure and gridspec needed for the pairwise comparison plot."""
        self._fig = plt.figure(figsize=(figsize[0]*self._nr_bstrps, 
                               figsize[1]*self._nr_bstrps))
        self._gs = gridspec.GridSpec(self._nr_bstrps, 3*self._nr_bstrps,
                        width_ratios=[wd/self._nr_bstrps for wd in _AX_WIDTH_RATIO]*self._nr_bstrps, 
                        height_ratios=[1/self._nr_bstrps]*self._nr_bstrps
                        )
        
    def _pairwise_logic_and_title(self, row: int, col: int):
        """
        Based on passed row/col of the pairwise plot, return the needed
        - Edge_logic instance(s)
        - Title
        """
        if row == col: # Precision case
            return [TP, FP], (self._titles[row]+_PREC_TITLE)
        elif col > row: # TP diff case
            return [TP_DIFF], (self._titles[row]+" to "+self._titles[col]+_TP_CHANGE_TITLE)
        elif col < row: # FP diff case
            return [FP_DIFF], (self._titles[row]+" to "+self._titles[col]+_FP_CHANGE_TITLE)
        else:
            raise ValueError("Invalid comparison with row/col indices.")

    def _pairwise_add_sortability(self, ax: matplotlib.axes.Axes, row: int, col: int):
        """Add sortability information to axes if needed for that row/col combination."""
        if row == col:
            lines = [
                f"Var Sort: {self._var_sortabilities[row]}",
                f"R2 Sort: {self._r2_sortabilities[row]}",
                f"Runtime: {str(timedelta(seconds=self._runtimes[row]))}",
                f"Invalid: {self._no_cons_extensions[row]}",
                f"Crashed: {self._alg_crashed[row]}"
            ]
            _text_top_left(
                axes=ax,
                txt="\n".join(lines)
                )
    
    #------------------------------------------------------
    # Private fcts for evolution plot

    def _setup_evolution_fig_and_grid(self, figsize: tuple):
        """Initializes the figure and gridspec needed for the evolution plot."""
        self._fig = plt.figure(figsize=(figsize[0]*3, 
                                        figsize[1]*(self._nr_bstrps+2)))
        self._gs = gridspec.GridSpec((self._nr_bstrps + 2), 9,
                               width_ratios=[wd/3 for wd in _AX_WIDTH_RATIO]*3)

    def _evolution_logic_graphs_title(self, ax: list[matplotlib.axes.Axes], row: int, col: int):
        """
        Based on passed row/col of the pairwise plot, return the needed
        - Edge_logic instance(s) for the plot
        - Reference graph
        - New graph
        - Title of the plot
        """
        # Determine which case we have
        true_case = self._true_dag_row_handler(ax, row, col) # Whitens ax if needed
        absolute_case = self._absolute_row_handler(row, col)
        relative_case = self._relative_row_handler(row, col)

        # Ensure that we have exactly one case
        not_none = [case is not None for case in (true_case, absolute_case, relative_case)]
        if not sum(not_none)==1:
            raise ValueError("Multiple or zero cases are correct for this row/col combination")
        
        if true_case is False: # No drawing required, axes has been whitened
            return False
        elif true_case is not None: # Draw true DAG case
            edge_logics, ref_graph, new_graph, title = true_case
        elif absolute_case is not None: # Draw absolute plot
            edge_logics, ref_graph, new_graph, title = absolute_case
        elif relative_case is not None: # Draw comparison plot
            edge_logics, ref_graph, new_graph, title = relative_case
        return edge_logics, ref_graph, new_graph, title

    def _true_dag_row_handler(self, ax: list[matplotlib.axes.Axes], row: int, col: int):
        """
        Handles the first row of the evolution plot, 
            where the true DAG shall be drawn in the third column.
        """
        if row == -1: 
            if col == _EV_COL[2]: # Draw True DAG
                edge_logics = [TRUE_EDGES]
                ref_graph = self._true_graph
                new_graph = self._true_graph
                title = _TRUE_TITLE
            elif col in _EV_COL[0:2]: # Draw nothing, whiten axes
                _whiten_axes(ax[0])
                _whiten_axes(ax[1])
                _whiten_axes(ax[2])
                return False # Flag that nothing shall be drawn
            else:
                raise ValueError("Col not known")
        else:
            return None # Flag that we are not in the true_dag row
        return edge_logics, ref_graph, new_graph, title

    def _absolute_row_handler(self, row: int, col: int):
        """
        Handles the second and last row of the evolution plot, 
            where an absolute graph shall be drawn per column.
        """
        if row == 0 or row == self._nr_bstrps: # Draw Absolute
            ind = 0 if row == 0 else -1 # First or last Graph/Title
            ref_graph = self._graphs[ind]
            new_graph = self._graphs[ind]
            if col == _EV_COL[0]: # TP 
                edge_logics = [TP]
                title = self._titles[ind]+_TP_TITLE
            elif col == _EV_COL[1]: # FP
                edge_logics = [FP] 
                title = self._titles[ind]+_FP_TITLE
            elif col == _EV_COL[2]: # Precision 
                edge_logics = [TP, FP]
                title = self._titles[ind]+_PREC_TITLE
            else:
                raise ValueError("Col not known")
        else:
            return None
        return edge_logics, ref_graph, new_graph, title
    
    def _relative_row_handler(self, row: int, col: int):
        """
        Handles the second and last row of the evolution plot, 
            where an absolute graph shall be drawn per column.
        """
        if row not in [-1, 0, self._nr_bstrps]: # Draw comparison
            ref_graph = self._graphs[row-1]
            new_graph = self._graphs[row]
            if col == _EV_COL[0]: # TP comp
                edge_logics = [TP_DIFF]
                title = self._titles[row-1]+" to "+self._titles[row]+_TP_CHANGE_TITLE
            elif col == _EV_COL[1]: # FP comp
                edge_logics = [FP_DIFF]
                title = self._titles[row-1]+" to "+self._titles[row]+_FP_CHANGE_TITLE
            elif col == _EV_COL[2]: # Precision 
                edge_logics = [TP, FP]
                ref_graph = self._graphs[row]
                new_graph = self._graphs[row]
                title = self._titles[row]+_PREC_TITLE
            else:
                raise ValueError("Col not known")
        else: 
            return None
        return edge_logics, ref_graph, new_graph, title
    

    def _evolutions_add_sortability(self, ax: matplotlib.axes.Axes, row: int, col: int):
        """Add sortability information to passed axes if col==3 and row!=-1."""
        if not row == -1 and col == 2:
            index = (self._nr_bstrps-1) if (row == self._nr_bstrps) else row
            lines = [
                f"Var Sort: {self._var_sortabilities[index]}",
                f"R2 Sort: {self._r2_sortabilities[index]}",
                f"Runtime: {str(timedelta(seconds=self._runtimes[index]))}",
                f"Invalid: {self._no_cons_extensions[index]}",
                f"Crashed: {self._alg_crashed[index]}"
            ]
            _text_top_left(
                axes=ax,
                txt="\n".join(lines)
                )


#------------------------------------------------------
# Module level helper fcts

def _vis(axes: Iterable[matplotlib.axes.Axes],
         graphs: AdjGraphs,
         pos: dict,
         latex_transf,
         edge_logics: list[EdgeLogic]):
    """
    Plots passed graphs onto the passed axes.

    Args:
        axes (Iterable[matplotlib.axes.Axes]): The axes to plot on.
        graphs (AdjGraphs): Graph object to plot.
        pos (dict): Node positions.
        latex_transf: Function transforming the variable names to 
            latex format that will then be used for the plot. 
        edge_logics (list[EdgeLogic]): List of edge_logics/types
            to put onto the axes.
    """
    # Create empty graph
    G = nx.DiGraph()

    # Add nodes
    nodes = Nodes(graphs=graphs,
                    pos=pos,
                    latex_transf=latex_transf)
    nodes.draw_nodes(G=G, ax_graph=axes[0])

    # Add desired edges
    for logic, ax in zip(edge_logics, axes[1:]):
        if logic in [TP_DIFF, FP_DIFF]:
            thres = _EDGE_THRESHOLD_COMP
        else:
            thres = _EDGE_THRESHOLD_ABS
        edges = Edges(graphs=graphs, logic=logic, threshold=thres)
        edges.draw_edges(G=G,
                            nodes=nodes,
                            ax_graph=axes[0],
                            ax_legend=ax)

    # Whiten non-used axeses
    delay = 0 if TRUE_EDGES in edge_logics else 1         
    for ax in axes[len(edge_logics)+delay:]:
        _whiten_axes(ax)
    
def _title(axes: matplotlib.axes.Axes, title: str):
    """Adds passed title tot he passed axes."""
    axes.set_title(label=title,
                   fontsize=_TITLE_FONTSIZE,
                   fontweight = _TITLE_FONTWEIGHT)
    
def _text_top_left(axes: matplotlib.axes.Axes, txt: str):
    """Adds passed text to the top-left corner of the passed axes."""
    axes.text(**_TEXT_POS,
              s=txt, 
              **_TEXT_ALIGN, 
              fontsize = _TEXT_FONTSIZE,
              transform = axes.transAxes)

def _whiten_axes(ax: matplotlib.axes.Axes) -> matplotlib.axes.Axes:
    """Turn the passed axes completely white by removing all elements on it."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    return ax

def _save_figure(fig: Figure, path: str):
    """Save passed figure under passed path."""
    try:
        fig.savefig(path, format="pdf", dpi=150, bbox_inches='tight')
        plt.clf()
        plt.close('all')
    except Exception as e:
        print(f"Failed to save the figure: {e}")
