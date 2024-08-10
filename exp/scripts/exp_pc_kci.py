from exp_assistant import (
    # Precreated datasets/true DAGs
    MID_VAR_TRUE_DAG,
    MID_VAR_UNIFORM_REFERENCE,
    # Other useful constants
    NR_BOOTSTRAPS,
    DEFAULT_DATA_SIZE,
    # Benchmarking classes
    BootstrapComparison,
    Bootstrap,
    # Algorithm 
    PC
)

### ------------------------------------------------------
# Set default configurations for PC algorithm
ALPHA = 0.1



def independence_test_comparison(processes: int = 50):
    bstrcomp = BootstrapComparison("PC-CiTestComparison")
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"FisherZ",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=PC(alpha=ALPHA, indep_test="fisherz"),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS, 
                PROCESSES=processes 
        )
    )
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"KCI",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=PC(alpha=ALPHA, indep_test="kci"),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES=processes 
        )
    )
    bstrcomp.run_comparison()
    bstrcomp.pickle()

if __name__ == "__main__":
    # Necessary trick: Run as many processes as bootstraps!
    independence_test_comparison(processes=NR_BOOTSTRAPS)