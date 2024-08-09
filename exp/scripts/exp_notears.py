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
    NoTears
)

# NoTEARs constants:
PROCESSES = 50
LAMBDA1 = 9


def dag_versus_cpdag_comparison(processes: int = 50):
    bstrcomp = BootstrapComparison("NoTears-DAGversusCPDAG")
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"Return DAG",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=NoTears(return_cpdag=False, lambda1=LAMBDA1),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS, 
                PROCESSES=processes
        )
    )
    bstrcomp.add_bootstrap(
        Bootstrap(
                name=f"Return CPDAG",
                true_dag=MID_VAR_TRUE_DAG,
                algorithm=NoTears(return_cpdag=True, lambda1=LAMBDA1),
                data_to_bootstrap_from=MID_VAR_UNIFORM_REFERENCE,
                sample_sizes=DEFAULT_DATA_SIZE,
                nr_bootstraps=NR_BOOTSTRAPS,
                PROCESSES=processes
        )
    )
    bstrcomp.run_comparison()
    bstrcomp.pickle()



if __name__ == "__main__":
    increase_obs_data_mid_var(alg=NoTears(lambda1=LAMBDA1), processes=PROCESSES)
    increase_variables(alg=NoTears(lambda1=LAMBDA1), processes=PROCESSES)
    increase_colors(alg=NoTears(lambda1=LAMBDA1), processes=PROCESSES)
    increase_hyperparameter(cls=NoTears, change_param_dict={'lambda1': [0.1, 1, 3, 9, 27, 81]}, processes=PROCESSES)
    standardized_data_comparison(alg=NoTears(lambda1=LAMBDA1), processes=PROCESSES)
    dag_versus_cpdag_comparison(processes=PROCESSES)

