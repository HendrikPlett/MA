from .assistant import (
    # Precreated datasets/true DAGs
    SMALL_VAR_TRUE_DAG,
    MID_VAR_TRUE_DAG,
    SMALL_VAR_UNIFORM_REFERENCE,
    MID_VAR_UNIFORM_REFERENCE,
    RED_TRUE_DAG,
    RED_GREEN_TRUE_DAG,
    RED_UNIFORM_REFERENCE,
    RED_GREEN_UNIFORM_REFERENCE,
    # Other useful constants
    NR_BOOTSTRAPS,
    OBS_DATA_COMP_SIZES,
    DEFAULT_DATA_SIZE,
    # Predefined benchmarking functions
    increase_obs_data_small_var,
    increase_obs_data_mid_var,
    increase_variables,
    increase_colors,
    increase_hyperparameter,
    # Benchmarking classes
    BootstrapComparison,
    Bootstrap,
    # Algorithm 
    PC
)

### ------------------------------------------------------
# Set default configurations for PC algorithm
ALPHA = 0.1

### ------------------------------------------------------
# Actual jobs

def independence_test_comparison():
    bstrcomp = BootstrapComparison("PC_CiTestComparison")
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"CI Test: FisherZ",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=PC(alpha=ALPHA, indep_test="fisherz"),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS        
        )
    )
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"CI Test: FisherZ",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=PC(alpha=ALPHA, indep_test="kci"),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS        
        )
    )

if __name__ == "__main__":
    increase_obs_data_small_var(alg=PC(alpha=ALPHA))
    increase_obs_data_mid_var(alg=PC(alpha=ALPHA))
    increase_variables(alg=PC(alpha=ALPHA))
    increase_colors(alg=PC(alpha=ALPHA))
    increase_hyperparameter(cls=PC, change_param_dict={'alpha': [0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6]})
    independence_test_comparison()

