import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(dir_path)
sys.path.append(src_path)

from causalbenchmark.compute.bootstrap import BootstrapComparison, Bootstrap
from causalbenchmark.compute.algorithms import Algorithm, PC, UT_IGSP, GES, GIES, GNIES, Golem, NoTears, ICP
from cc_wrapper import CCWrapper
from cc_wrapper import SMALL_VAR, MID_VAR, ALL_VAR # Variable subsets

### ------------------------------------------------------
#
# Some constants used frequently for comparisons
#
### ------------------------------------------------------

CLUSTER_CPUS = 51
NR_BOOTSTRAPS = 100 # Change to 100 later on for MasterThesis
OBS_DATA_COMP_SIZES = [50, 125, 250, 500, 1000, 2000, 4000, 10000]
DEFAULT_DATA_SIZE = [1000]

### ------------------------------------------------------
#
# Get some frequently used CausalChamber observational data
#
### ------------------------------------------------------

### ------------------------------------------------------
# 1) lt_interventions, uniform reference, predefined variable sets
_EXP_FAMILY = "lt_interventions_standard_v1"
_EXPERIMENT = ["uniform_reference"]

_ccw = CCWrapper()
_ccw.set_exp_family(_EXP_FAMILY)


_ccw.set_variables(SMALL_VAR)
_SMALL_VAR_TRUE_DAG = _ccw.fetch_true_dag()
_SMALL_VAR_UNIFORM_REFERENCE = _ccw.fetch_experiments(
                    experiments=_EXPERIMENT
                    ) # Returns list

_ccw.set_variables(MID_VAR)
_MID_VAR_TRUE_DAG = _ccw.fetch_true_dag()
_MID_VAR_UNIFORM_REFERENCE = _ccw.fetch_experiments(
                    experiments=_EXPERIMENT
                    ) # Returns list

_ccw.set_variables(ALL_VAR)
_ALL_VAR_TRUE_DAG = _ccw.fetch_true_dag()
_ALL_VAR_UNIFORM_REFERENCE = _ccw.fetch_experiments(
                    experiments=_EXPERIMENT
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
# 2) lt_interventions, uniform reference, drop colors

_ccw.set_variables([var for var in MID_VAR if var not in ['green', 'blue']]) # Drop variables 'green' and 'blue'
_RED_TRUE_DAG = _ccw.fetch_true_dag()
_RED_UNIFORM_REFERNECE = _ccw.fetch_experiments(
                experiments=_EXPERIMENT
            )

_ccw.set_variables([var for var in MID_VAR if var not in ['blue']]) # Drop variable 'blue'
_RED_GREEN_TRUE_DAG = _ccw.fetch_true_dag()
_RED_GREEN_UNIFORM_REFERNECE = _ccw.fetch_experiments(
                experiments=_EXPERIMENT
            )

# --- Public API ---
RED_TRUE_DAG = _RED_TRUE_DAG
RED_UNIFORM_REFERENCE = _RED_UNIFORM_REFERNECE
RED_GREEN_TRUE_DAG = _RED_GREEN_TRUE_DAG
RED_GREEN_UNIFORM_REFERENCE = _RED_GREEN_UNIFORM_REFERNECE
# --- Public API ---



### ------------------------------------------------------
#
# Get some frequently used CausalChamber interventional data
#
### ------------------------------------------------------


# TODO: Fill this
# Define INTERVENTIONS = [list of variables]
# Define DATASET = [list of the datasets with the respective interventions]


### ------------------------------------------------------
# 
# Define functions often used for benchmarks - first on observational data
#
### ------------------------------------------------------


def increase_obs_data_small_var(alg: Algorithm):
    prefix = alg.__class__.__name__
    bstrpcomp = BootstrapComparison(f"{prefix}-SmallVar-IncreaseObservationalData")
    for size in OBS_DATA_COMP_SIZES:
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"n = {size}",
                true_dag=SMALL_VAR_TRUE_DAG,
                algorithm=alg,
                data_to_bootstrap_from=SMALL_VAR_UNIFORM_REFERENCE,
                sample_sizes=[size],
                nr_bootstraps=NR_BOOTSTRAPS,
                CLUSTER_CPUS=CLUSTER_CPUS
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()


def increase_obs_data_mid_var(alg: Algorithm):
    prefix = alg.__class__.__name__
    bstrpcomp = BootstrapComparison(f"{prefix}-MidVar-IncreaseObservationalData")
    for size in OBS_DATA_COMP_SIZES:
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"n = {size}",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=alg,
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=[size],
                nr_bootstraps=NR_BOOTSTRAPS,
                CLUSTER_CPUS=CLUSTER_CPUS
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()

def increase_variables(alg: Algorithm):
    names = ['Small Var', 'Med Var']
    true_dags = [SMALL_VAR_TRUE_DAG, MID_VAR_TRUE_DAG]
    datas = [SMALL_VAR_UNIFORM_REFERENCE, MID_VAR_UNIFORM_REFERENCE]
    prefix = alg.__class__.__name__
    bstrpcomp = BootstrapComparison(f"{prefix}-IncreaseVariableCount")
    for name, true_dag, data in zip(names, true_dags, datas):
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=name,
                true_dag=true_dag,
                algorithm=alg,
                data_to_bootstrap_from=data,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS, 
                CLUSTER_CPUS=CLUSTER_CPUS
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()


def increase_colors(alg: Algorithm):
    names = ['Red', 'Red, Green', 'Red, Green, Blue']
    true_dags = [RED_TRUE_DAG, RED_GREEN_TRUE_DAG, MID_VAR_TRUE_DAG]
    datas = [RED_UNIFORM_REFERENCE, RED_GREEN_UNIFORM_REFERENCE, MID_VAR_UNIFORM_REFERENCE]
    prefix = alg.__class__.__name__
    bstrpcomp = BootstrapComparison(f"{prefix}-IncreaseColorCount")
    for name, true_dag, data in zip(names, true_dags, datas):
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=name,
                true_dag=true_dag,
                algorithm=alg,
                data_to_bootstrap_from=data,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS, 
                CLUSTER_CPUS=CLUSTER_CPUS
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()


def _instantiate_class(cls, change_param_dict: dict, fix_param_dict: dict = {}) -> list:
    instances = []
    # Change_param_dict only has a single key, value pair -> Retrive it
    change_key, change_values = next(iter(change_param_dict.items()))
    
    # Iterate over each parameter in the change_values list
    for value in change_values:
        # Create dictionary with the fixed parameters and the current flexible parameter
        all_params = fix_param_dict.copy()
        all_params[change_key] = value
        # Instantiate class with combined parameters
        instance = cls(**all_params)
        instances.append(instance)
    
    return instances


def increase_hyperparameter(cls, change_param_dict: dict, fix_param_dict: dict = {}):
    hyperparm = list(change_param_dict.keys())[0]
    bstrpcomp = BootstrapComparison(f"{cls.__name__}-Increase{hyperparm.capitalize()}")
    alg_instances = _instantiate_class(cls=cls, 
                                       change_param_dict=change_param_dict, 
                                       fix_param_dict=fix_param_dict)
    for alg in alg_instances:
        value = getattr(alg, f"_{hyperparm}")
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"{hyperparm.capitalize()}: {value}",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=alg,
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS, 
                CLUSTER_CPUS=CLUSTER_CPUS
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()


def standardized_data_comparison(alg: Algorithm):
    bstrpcomp = BootstrapComparison(f"{alg.__class__.__name__}-StandardizeComparison")
    bstrpcomp.add_bootstrap(
        Bootstrap(
            name=f"Original Scale",
            true_dag=MID_VAR_TRUE_DAG,
            algorithm=alg,
            data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
            sample_sizes=DEFAULT_DATA_SIZE,
            standardize_data=False,
            nr_bootstraps=NR_BOOTSTRAPS,
            CLUSTER_CPUS=CLUSTER_CPUS
        )
    )
    bstrpcomp.add_bootstrap(
        Bootstrap(
            name=f"Standardized Scale",
            true_dag=MID_VAR_TRUE_DAG,
            algorithm=alg,
            data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
            sample_sizes=DEFAULT_DATA_SIZE,
            standardize_data=True,
            nr_bootstraps=NR_BOOTSTRAPS, 
            CLUSTER_CPUS=CLUSTER_CPUS
        )
    )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()

