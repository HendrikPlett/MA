import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(dir_path)
sys.path.append(src_path)

from causalbenchmark.compute.bootstrap import BootstrapComparison, Bootstrap
from causalbenchmark.compute.algorithms import PC
from cc_wrapper import CCWrapper
from cc_wrapper import SMALL_VAR, MID_VAR, ALL_VAR, LATEX_NAME


### ------------------------------------------------------
# Get CausalChamber data

ccw = CCWrapper()
ccw.set_exp_family("lt_interventions_standard_v1")
ccw.set_variables(SMALL_VAR)
# Fetch True DAG
SMALL_VAR_TRUE_DAG = ccw.fetch_true_dag()
# Fetch data
SMALL_VAR_UNIFORM_REFERENCE = ccw.fetch_experiments(
                    experiments=['uniform_reference'], 
                    sizes=[10000]
                    ) # Returns list

ccw.set_variables(MID_VAR)
# Fetch True DAG
MID_VAR_TRUE_DAG = ccw.fetch_true_dag()
# Fetch data
MID_VAR_UNIFORM_REFERENCE = ccw.fetch_experiments(
                    experiments=['uniform_reference'], 
                    sizes=[10000]
                    ) # Returns list

ccw.set_variables(ALL_VAR)
# Fetch True DAG
ALL_VAR_TRUE_DAG = ccw.fetch_true_dag()
# Fetch data
ALL_VAR_UNIFORM_REFERENCE = ccw.fetch_experiments(
                    experiments=['uniform_reference'], 
                    sizes=[10000]
                    ) # Returns list
