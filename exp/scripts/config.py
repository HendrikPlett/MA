import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(dir_path)
sys.path.append(src_path)

from causalbenchmark.compute.bootstrap import BootstrapComparison, Bootstrap
from causalbenchmark.compute.algorithms import PC
from cc_wrapper import CCWrapper
from cc_wrapper import SMALL_VAR, MID_VAR, ALL_VAR

### ------------------------------------------------------
# Get some frequently used CausalChamber data
### ------------------------------------------------------


### ------------------------------------------------------
# lt_interventions, uniform reference, predefined variable sets
_EXP_FAMILY = "lt_interventions_standard_v1"
_EXPERIMENT = ["uniform_reference"]
_SIZE = [1000]

_ccw = CCWrapper()
_ccw.set_exp_family(_EXP_FAMILY)
_ccw.set_variables(SMALL_VAR)
# Fetch True DAG
_SMALL_VAR_TRUE_DAG = _ccw.fetch_true_dag()
# Fetch data
_SMALL_VAR_UNIFORM_REFERENCE = _ccw.fetch_experiments(
                    experiments=_EXPERIMENT, 
                    sizes=_SIZE
                    ) # Returns list

_ccw.set_variables(MID_VAR)
# Fetch True DAG
_MID_VAR_TRUE_DAG = _ccw.fetch_true_dag()
# Fetch data
_MID_VAR_UNIFORM_REFERENCE = _ccw.fetch_experiments(
                    experiments=_EXPERIMENT, 
                    sizes=_SIZE
                    ) # Returns list

_ccw.set_variables(ALL_VAR)
# Fetch True DAG
_ALL_VAR_TRUE_DAG = _ccw.fetch_true_dag()
# Fetch data
_ALL_VAR_UNIFORM_REFERENCE = _ccw.fetch_experiments(
                    experiments=_EXPERIMENT, 
                    sizes=_SIZE
                    ) # Returns list


# --- Public API ---
SMALL_VAR_TRUE_DAG = _SMALL_VAR_TRUE_DAG
SMALL_VAR_UNIFORM_REFERENCE = _SMALL_VAR_UNIFORM_REFERENCE
MID_VAR_TRUE_DAG = _MID_VAR_TRUE_DAG
MID_VAR_UNIFORM_REFERENCE = _MID_VAR_UNIFORM_REFERENCE
ALL_VAR_TRUE_DAG = _ALL_VAR_TRUE_DAG
ALL_VAR_UNIFORM_REFERENCE = _ALL_VAR_UNIFORM_REFERENCE
# --- Public API ---


### ------------------------------------------------------
# lt_interventions, uniform reference, drop colors

_ccw.set_variables([var for var in MID_VAR if var not in ['green', 'blue']])
_RED_TRUE_DAG = _ccw.fetch_true_dag()
_RED_UNIFORM_REFERNECE = _ccw.fetch_experiments(
                experiments=_EXPERIMENT,
                sizes=_SIZE
            )

_ccw.set_variables([var for var in MID_VAR if var not in ['blue']])
_RED_GREEN_TRUE_DAG = _ccw.fetch_true_dag()
_RED_GREEN_UNIFORM_REFERNECE = _ccw.fetch_experiments(
                experiments=_EXPERIMENT,
                sizes=_SIZE
            )

# --- Public API ---
RED_TRUE_DAG = _RED_TRUE_DAG
RED_UNIFORM_REFERENCE = _RED_UNIFORM_REFERNECE
RED_GREEN_TRUE_DAG = _RED_GREEN_TRUE_DAG
RED_GREEN_UNIFORM_REFERENCE = _RED_GREEN_UNIFORM_REFERNECE
# --- Public API ---
