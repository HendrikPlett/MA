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
    increase_hyperparameter,
    standardized_data_comparison,
    # Benchmarking classes
    BootstrapComparison,
    Bootstrap,
    # Algorithm 
    PC
)

### ------------------------------------------------------
# Set default configurations for PC algorithm
ALPHA = 0.1



if __name__ == "__main__":
    increase_obs_data_small_var(alg=PC(alpha=ALPHA))
    increase_obs_data_mid_var(alg=PC(alpha=ALPHA))
    increase_variables(alg=PC(alpha=ALPHA))
    increase_colors(alg=PC(alpha=ALPHA))
    increase_hyperparameter(cls=PC, change_param_dict={'alpha': [0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6]})
    standardized_data_comparison(alg=PC(alpha=ALPHA))

