from operator import gt, eq, abs 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from  matplotlib.colors import LinearSegmentedColormap



_ZEROONE = mcolors.Normalize(vmin=0, vmax=1)
_MINUSONEONE = mcolors.Normalize(vmin=-1, vmax=1)

def true(a,b):
    return True

def abs_gt(a,b):
    return gt(abs(a), b)

class EdgeLogic:

    def __init__(self,
                 graph_comp, 
                 true_graph_comp,
                 colormap,
                 normalizer
                 ):
        
        self._graph_comp = graph_comp
        self._true_graph_comp = true_graph_comp
        self._colormap = colormap
        self._normalizer = normalizer

    @property
    def graph_comp(self):
        return self._graph_comp
    
    @property
    def true_graph_comp(self):
        return self._true_graph_comp
    
    @property
    def colormap(self):
        return self._colormap
    
    @property
    def normalizer(self):
        return self._normalizer

_all_p = EdgeLogic(graph_comp=gt,
                 true_graph_comp = true,
                 colormap=plt.cm.Greys,
                 normalizer=_ZEROONE)    

_tp = EdgeLogic(graph_comp=gt,
               true_graph_comp=gt,
               colormap=plt.cm.Greens, 
               normalizer=_ZEROONE)

_fp = EdgeLogic(graph_comp=gt,
               true_graph_comp=eq,
               colormap=plt.cm.Reds, 
               normalizer=_ZEROONE)

_tp_diff = EdgeLogic(graph_comp=abs_gt,
                     true_graph_comp=gt,
                     colormap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256),
                     normalizer=_MINUSONEONE)

_fp_diff = EdgeLogic(graph_comp=abs_gt,
                     true_graph_comp=eq,
                     colormap=LinearSegmentedColormap.from_list('rg',["g", "w", "r"], N=256),
                     normalizer=_MINUSONEONE)

###
# Public API

all_p = _all_p
tp = _tp
fp = _fp
tp_diff = _tp_diff
fp_diff = _fp_diff




