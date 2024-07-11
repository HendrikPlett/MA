"""
Module defining the logic needed to compute various edges 
    types, for example True Positive, False Positive edges.
"""

from typing import Callable
from operator import gt, eq, abs 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from  matplotlib.colors import LinearSegmentedColormap

_COLMAP_ALL = plt.cm.Greys 
_COLMAP_TP = plt.cm.Greens
_COLMAP_FP = plt.cm.Reds
_COLMAP_TP_DIFF = LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)
_COLMAP_FP_DIFF = LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256)

_ZEROONE = mcolors.Normalize(vmin=0, vmax=1)
_MINUSONEONE = mcolors.Normalize(vmin=-1, vmax=1)

def true(a,b):
    """= True for all possible a,b."""
    return True

def abs_gt(a,b):
    """Whether |a|>b."""
    return gt(abs(a), b)

class EdgeLogic:
    """Allows to define when and in what color an edge is drawn."""
    def __init__(self,
                 graph_comp: Callable[[float, float], bool], 
                 true_graph_comp: Callable[[float, float], bool],
                 colormap,
                 normalizer: mcolors.Normalize,
                 label: str
                 ):
        """
        Passing values to be stored as constants.

        Args:
            graph_comp (Callable[[float, float], bool]): Function 
                taking a value from the graph (first input) that shall be 
                plotted and comparing this value to a defined threshold (second
                input). If the function evaluates to true (and the true_graph_comp
                function) as well, the edge corresponding to that value will be 
                drawn.
            true_graph_comp (Callable[[float, float], bool]): Function 
                taking a value from the true graph (first input) and comparing this 
                value to a defined threshold (second input). If the function evaluates 
                to true (and the graph_comp function) as well, the edge corresponding
                to the graph_comp input will be drawn.
            colormap (Any): The colormap to use for a certain edge type.
            normalizer (mcolors.Normalize): Mapping from the range of the first input
                of the graph_comp function to [0,1].
            label (str): The label used to describe that edge type in the final plot.
        """
        self._graph_comp = graph_comp
        self._true_graph_comp = true_graph_comp
        self._colormap = colormap
        self._normalizer = normalizer
        self._label = label

    @property
    def graph_comp(self) -> Callable[[float, float], bool]:
        """
        Getting the function that must be applied to the 
            graph one wants to plot.
        """
        return self._graph_comp
    
    @property
    def true_graph_comp(self) -> Callable[[float, float], bool]:
        """Getting the function that must be applied to the true DAG."""
        return self._true_graph_comp
    
    @property
    def colormap(self):
        """Get the colormap of this EdgeType instance."""
        return self._colormap
    
    @property
    def normalizer(self) -> mcolors.Normalize:
        """Get the normalizer of this EdgeType instance."""
        return self._normalizer
    
    @property
    def label(self) -> str:
        """Get the label/legend description of this EdgeType instance."""
        return self._label


#------------------------------------------------------
# Define EdgeTypes

_true_edges = EdgeLogic(graph_comp=gt,
                        true_graph_comp= true,
                        colormap=_COLMAP_ALL,
                        normalizer=_ZEROONE,
                        label=None)

_all_p = EdgeLogic(graph_comp=gt,
                 true_graph_comp = true,
                 colormap=_COLMAP_ALL,
                 normalizer=_ZEROONE,
                 label='Edge discovered in %')    

_tp = EdgeLogic(graph_comp=gt,
               true_graph_comp=gt,
               colormap=_COLMAP_TP, 
               normalizer=_ZEROONE,
               label='True Positive in %')

_fp = EdgeLogic(graph_comp=gt,
               true_graph_comp=eq,
               colormap=_COLMAP_FP, 
               normalizer=_ZEROONE,
               label='False Positive in %')

_tp_diff = EdgeLogic(graph_comp=abs_gt,
                     true_graph_comp=gt,
                     colormap=_COLMAP_TP_DIFF,
                     normalizer=_MINUSONEONE,
                     label="True positives: Change in %pts")

_fp_diff = EdgeLogic(graph_comp=abs_gt,
                     true_graph_comp=eq,
                     colormap=_COLMAP_FP_DIFF,
                     normalizer=_MINUSONEONE,
                     label="False positives: Change in %pts")


#------------------------------------------------------
# Public API
TRUE_EDGES = _true_edges
ALL_P = _all_p
TP = _tp
FP = _fp
TP_DIFF = _tp_diff
FP_DIFF = _fp_diff
