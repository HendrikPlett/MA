import sys
import os

dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src'))
sys.path.append(dir_path)
sys.path.append(src_path)

from causalbenchmark.compute.bootstrap import BootstrapComparison, Bootstrap
from causalbenchmark.compute.algorithms import PC
from cc_wrapper import CCWrapper
from cc_wrapper import CORE_VAR, WO_PARAM_VAR, POSITIONS_LT, LATEX_NAME


def main():
    # Set up the CausalChamberWrapper
    ccw = CCWrapper()
    ccw.set_exp_family("lt_interventions_standard_v1")
    ccw.set_variables(CORE_VAR)
    # Fetch True DAG
    true_dag = ccw.fetch_true_dag()
    # Fetch data
    data = ccw.fetch_experiments(experiments=['uniform_reference'], sizes=[10000])[0] # Returns list, fetch first element
    bc = BootstrapComparison(name="First_Comparison")
    bc.add_bootstrap(
        Bootstrap(
            name="EULER1",
            algorithm=PC(alpha=0.05),
            data_to_bootstrap_from=[data],
            sample_sizes=(10000,),
            nr_bootstraps=30,
            true_dag=true_dag
        )
    )
    bc.add_bootstrap(
        Bootstrap(
            name="EULER2",
            algorithm=PC(alpha=0.05),
            data_to_bootstrap_from=[data],
            sample_sizes=(10000,),
            nr_bootstraps=30,
            true_dag=true_dag
        )
    )
    bc.run_comparison()
    bc.pickle()


if __name__ == "__main__":
    main()