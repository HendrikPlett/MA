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
# Set default configurations for PC algorithm
SAMPLE_SIZE = 1000
ALPHA = 0.05


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



### ------------------------------------------------------
# Actual jobs

def increase_obs_data_small_var(sizes: list[int]):
    bstrpcomp = BootstrapComparison("PC-SmallVar-IncreaseObservationalData")
    for size in sizes:
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"n = {size}",
                true_dag=SMALL_VAR_TRUE_DAG,
                algorithm=PC(alpha=ALPHA),
                data_to_bootstrap_from=SMALL_VAR_UNIFORM_REFERENCE,
                sample_sizes=[size],
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()


def increase_obs_data_mid_var(sizes: list[int]):
    bstrpcomp = BootstrapComparison("PC-MidVar-IncreaseObservationalData")
    for size in sizes:
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"n = {size}",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=PC(alpha=ALPHA),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=[size],
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()

def increase_variables():
    names = ['Few Variables', 'Medium Variables', 'All Variables']
    true_dags = [SMALL_VAR_TRUE_DAG, MID_VAR_TRUE_DAG, ALL_VAR_TRUE_DAG]
    datas = [SMALL_VAR_UNIFORM_REFERENCE, MID_VAR_UNIFORM_REFERENCE, ALL_VAR_UNIFORM_REFERENCE]
    bstrpcomp = BootstrapComparison("PC-IncreaseVariableCount")
    for name, true_dag, data in zip(names, true_dags, datas):
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=name,
                true_dag=true_dag,
                algorithm=PC(alpha=ALPHA),
                data_to_bootstrap_from=data,
                sample_sizes=[SAMPLE_SIZE],
            )
        )
    bstrpcomp.run_comparison()
    bstrpcomp.pickle()

def increase_alpha(alphas: list):
    bstrpcomp = BootstrapComparison("PC-IncreaseAlpha")
    for alpha in alphas:
        bstrpcomp.add_bootstrap(
            Bootstrap(
                name=f"Alpha: {alpha}",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=PC(alpha=alpha),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=[SAMPLE_SIZE],
            )
        )


if __name__ == "__main__":
    increase_obs_data_small_var(sizes=[100, 200, 400, 800, 1600, 3200, 6400])
    increase_obs_data_mid_var(sizes=[100, 200, 400, 800, 1600, 3200, 6400])
    increase_variables()
    increase_alpha(alphas=[0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6])

