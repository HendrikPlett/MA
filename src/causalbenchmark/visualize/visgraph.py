
import matplotlib.axes
import networkx

from .helper import AdjGraphs
from .edges import Edges
from .nodes import Nodes

class VisGraph:
    

    def __init__(self, 
                 graphs: AdjGraphs,
                 title: str, 
                 axeses: list[matplotlib.axes.Axes]
                 ):
        
        # --- Provided
        self._graphs = graphs
        self._title = title
        self._ax1, self._ax2, self._ax3 = axeses
        # --- Computed later 
        # Nodes/Variables
        self._nds = None
        self._core_var, self._diff_var, self._rest_var = [], [], []
        self._core_col, self._diff_col, self._rest_col = [], [], []
        # Edges
        self._edgs = None
        self._edges, self._edge_colors = [], []

    #------------------------------------------------------
    # Computing nodes and edges

    def comp_all_edges(self):
        self._init_edges()
        self._edgs.comp_all()
        self._finalize_edges()

    def comp_tp_edges(self):
        self._init_edges()
        self._edgs.comp_tp()
        self._finalize_edges()

    def comp_fp_edges(self):
        self._init_edges()
        self._edgs.comp_all()
        self._finalize_edges()

    def comp_tp_diff_edges(self):
        self._init_edges()
        self._edgs.comp_all()
        self._finalize_edges()

    def comp_fp_diff_edges(self):
        self._init_edges()
        self._edgs.comp_all()
        self._finalize_edges()

    def comp_nodes(self):
        nds = Nodes(self._graphs)
        nds.compute_var_groups()
        self._core_var, self._diff_var, self._rest_var = nds.var_split
        self._core_col, self._diff_col, self._rest_col = nds.var_cols

    #------------------------------------------------------
    # Drawing nodes/edges onto the passed axeses


    #------------------------------------------------------
    # Private
    
    def _init_edges(self):
        self._edgs = Edges(self._graphs)

    def _finalize_edges(self):
        self._edges, self._edge_colors = self._edgs.edgesandcolors

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

