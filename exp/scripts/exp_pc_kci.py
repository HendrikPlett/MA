from exp_assistant import (
    # Precreated datasets/true DAGs
    MID_VAR_TRUE_DAG,
    MID_VAR_UNIFORM_REFERENCE,
    # Other useful constants
    CLUSTER_CPUS,
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


def independence_test_comparison():
    bstrcomp = BootstrapComparison("PC_CiTestComparison")
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"CI Test: FisherZ",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=PC(alpha=ALPHA, indep_test="fisherz"),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=5, 
                CLUSTER_CPUS=False # Run sequentially      
        )
    )
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"CI Test: KCI",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=PC(alpha=ALPHA, indep_test="kci"),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=5,
                CLUSTER_CPUS=False # Run sequentially, no parallelization possible apparently for "kci"
        )
    )
    bstrcomp.run_comparison()
    bstrcomp.pickle()

if __name__ == "__main__":
    independence_test_comparison()