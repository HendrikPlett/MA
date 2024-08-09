from exp_assistant import (
    # Precreated datasets/true DAGs
    MID_VAR_TRUE_DAG,
    MID_VAR_UNIFORM_REFERENCE,
    # Other useful constants
    NR_BOOTSTRAPS,
    DEFAULT_DATA_SIZE,
    # Predefined benchmarking functions
    increase_obs_data_small_var,
    increase_obs_data_mid_var,
    increase_variables,
    increase_colors,
    standardized_data_comparison,
    # Benchmarking classes
    BootstrapComparison,
    Bootstrap,
    # Algorithm 
    GES
)


### ------------------------------------------------------
# Experiments specific to this algorithm will be defined here

if __name__ == "__main__":
    increase_obs_data_small_var(alg=GES())
    increase_obs_data_mid_var(alg=GES())
    increase_variables(alg=GES())
    increase_colors(alg=GES())
    standardized_data_comparison(alg=GES())

